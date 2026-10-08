"""
boe_monitor.py — Monitor de publicaciones del BOE con verificación jurídica profunda.

Detecta nuevas normas que puedan hacer obsoletos los PDFs indexados en Tambora.

Flujo por documento:
  1. _classify_document(): routing — UNE_ISO → catálogos AENOR/ISO; BOE → API BOE
  2. Filtro de jurisdicción: excluye legislación autonómica/foral (_is_state_norm)
  3. Stage 1: keyword debe aparecer en TÍTULO del documento BOE
  4. Stage 2: fetch_boe_sections() → análisis por zona jurídica:
       disposición derogatoria + ref_term → derogado_total
       artículos/disposición final + ref_term + término modificador → modificado_parcialmente
       preámbulo/texto completo + ref_term → referenciado_sin_efecto
  5. Resultado con campo evidencia: zona, artículos afectados, texto literal

Uso manual:  python src/chatbot/boe_monitor.py (desde la raiz del modulo)
Programado:  schtasks (ver instrucciones al final del archivo)
"""

import json
import re
import sys
import time
import unicodedata
from datetime import date, datetime
from pathlib import Path

import httpx

# Este fichero vive en src/chatbot/, asi que parents[2] es la raiz del modulo.
# Se anade src/ al sys.path para poder ejecutarlo tambien como script suelto
# (python src/chatbot/<este fichero>.py), no solo importado por la app.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from chatbot.core.boe_search import BOESearchClient  # noqa: E402 (tras preparar sys.path)
from chatbot.core.pdf_loader import PDFLoader  # noqa: E402 (tras preparar sys.path)

# Try to import normative_tracker modules (available after v1.3.0 deployment)
try:
    from chatbot.core.normative_tracker.identity_extractor import (
        classify_document_origin,
        parse_une_iso_identifier,
    )
    from chatbot.core.normative_tracker.legal_parser import (
        parse_legal_terms,  # noqa: F401 (parte de la comprobación de que el módulo existe)
        extract_affected_articles,
    )

    _TRACKER_AVAILABLE = True
except ImportError:
    _TRACKER_AVAILABLE = False

try:
    from chatbot.core.normative_tracker.une_iso_checker import UNEISOChecker

    _UNE_ISO_CHECKER_AVAILABLE = True
except ImportError:
    _UNE_ISO_CHECKER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ALERTS_FILE = _ROOT / "data" / "doc_alerts.json"
STATUS_FILE = _ROOT / "data" / "check_status.json"

# ---------------------------------------------------------------------------
# Spanish stopwords for keyword extraction
# ---------------------------------------------------------------------------
_STOPS = {
    "real",
    "decreto",
    "orden",
    "ministerial",
    "ley",
    "organica",
    "sobre",
    "para",
    "entre",
    "contra",
    "segun",
    "mediante",
    "espanol",
    "espanola",
    "national",
    "norma",
    "normas",
    "parte",
    "titulo",
    "capitulo",
    "anexo",
    "version",
    "criterios",
    "guia",
    "tecnica",
    "evaluacion",
    "prevencion",
    "riesgos",
    "relacionados",
    "proteccion",
    "frente",
    "riesgo",
}

_KNOWN_ACRONYMS = {"cae", "rite", "rebt", "idae", "cnmc", "ree", "miteco"}

# ---------------------------------------------------------------------------
# Jurisdiction filter — exclude regional/autonomous community legislation
# ---------------------------------------------------------------------------
_REGIONAL_DEPT_RE = re.compile(
    r"junta\s+de|generalitat|govern\s+de|xunta|gobierno\s+foral"
    r"|diputacion\s+foral|comunitat|eusko\s+jaurlaritza|consell\s+insular"
    r"|comunidad\s+autonoma|region\s+de\s+murcia|govern\s+balear",
    re.I,
)

# ---------------------------------------------------------------------------
# BOE section headers for section-aware parsing
# ---------------------------------------------------------------------------
_SECTION_HEADERS = {
    "disp_derogatorias": re.compile(r"disposici[oó]n\s+derogator", re.I),
    "disp_finales": re.compile(r"disposici[oó]n\s+final", re.I),
    "disp_adicionales": re.compile(r"disposici[oó]n\s+adicional", re.I),
    "disp_transitorias": re.compile(r"disposici[oó]n\s+transit", re.I),
    "articulos": re.compile(r"art[ií]culo\s+\d", re.I),
}

# Derogation / modification patterns (applied to normalized ASCII text)
_DEROGATE = re.compile(
    r"deroga|anula\b|deja sin efecto|queda derogad|quedan derogad|pierde vigencia", re.I
)
_MODIFY = re.compile(
    r"modifica|nueva redacci[oó]n|se da nueva|queda redactado|a[nñ]ade|a[nñ]adi[eé]ndose"
    r"|sustitu|nueva versi[oó]n|nueva edici[oó]n|reemplaz",
    re.I,
)
_SUBSTITUTE = re.compile(r"sustitu|nueva versi|nueva edici|reemplaz", re.I)

# BOE identifier pattern in URLs
_BOE_ID_RE = re.compile(r"BOE-[A-Z]-\d{4}-\d+", re.I)


def _normalize(text: str) -> str:
    """Lowercase + remove accents + ASCII only."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return nfkd.encode("ascii", "ignore").decode("ascii")


# ---------------------------------------------------------------------------
# Status metadata
# ---------------------------------------------------------------------------
_RESULTADO_TEXT = {
    "en_vigor": "Verificado en vigor. No se detectaron publicaciones que modifiquen este documento.",
    "posible_cambio": "Se encontraron publicaciones BOE relacionadas por palabras clave en título, pero sin referencia textual directa al documento.",
    "referenciado_sin_efecto": "El documento es mencionado en el preámbulo/considerandos de una publicación BOE posterior, pero sin efecto jurídico operativo.",
    "modificado_parcialmente": "Una publicación BOE posterior contiene referencias al documento en artículos o disposiciones finales con efecto modificador confirmado.",
    "derogado_total": "Una disposición derogatoria de una publicación BOE posterior hace referencia explícita a este documento.",
    "sustituido": "Una publicación BOE posterior sustituye explícitamente este documento.",
    "no_encontrado": "No se encontraron coincidencias en la legislación consolidada del BOE.",
    # legacy states
    "derogado": "Una publicación BOE posterior deroga explícitamente este documento.",
    "modificado": "Se encontraron publicaciones más recientes que pueden modificar este documento.",
    "referencia_directa": "Una publicación BOE posterior contiene referencias directas a este documento.",
    "error": "Error durante la consulta a la API del BOE.",
}

_RECOMENDACION_TEXT = {
    "en_vigor": "Mantener el documento indexado. No se requiere acción.",
    "posible_cambio": "Revisar manualmente las publicaciones detectadas para confirmar si afectan al documento.",
    "referenciado_sin_efecto": "Sin acción inmediata necesaria. La mención es meramente referencial, sin efecto modificador.",
    "modificado_parcialmente": "Revisar el documento BOE detectado y actualizar los artículos afectados en el indexado.",
    "derogado_total": "Marcar el documento como no vigente y excluirlo de consultas futuras.",
    "sustituido": "Reemplazar el documento indexado por la versión vigente indicada.",
    "no_encontrado": "Verificar manualmente en el BOE. No se encontraron coincidencias automáticas.",
    # legacy
    "derogado": "Marcar el documento como no vigente.",
    "modificado": "Revisar y actualizar el documento indexado.",
    "referencia_directa": "Revisar el documento BOE detectado y actualizar si procede.",
    "error": "Reintentar la verificación o revisar manualmente.",
}

_ACCION = {
    "en_vigor": "mantener",
    "posible_cambio": "revision_manual",
    "referenciado_sin_efecto": "monitorizar",
    "modificado_parcialmente": "actualizar",
    "derogado_total": "desindexar",
    "sustituido": "actualizar",
    "no_encontrado": "revision_manual",
    "derogado": "desindexar",
    "modificado": "actualizar",
    "referencia_directa": "actualizar",
    "error": "reintentar",
}

_NIVEL_ALERTA = {
    "en_vigor": "verde",
    "posible_cambio": "amarillo",
    "referenciado_sin_efecto": "amarillo",
    "modificado_parcialmente": "naranja",
    "derogado_total": "rojo",
    "sustituido": "rojo",
    "no_encontrado": "gris",
    "derogado": "rojo",
    "modificado": "amarillo",
    "referencia_directa": "naranja",
    "error": "gris",
}


# ---------------------------------------------------------------------------
# Keyword extraction from PDF filename
# ---------------------------------------------------------------------------


def extract_keywords(filename: str) -> list:
    """
    Extract searchable keywords from a PDF filename stem.
    Returns 4+ digit sequences (norm numbers / years) and significant content
    words (>4 chars) after normalising accents and splitting on separators.
    """
    stem = Path(filename).stem
    norm = _normalize(stem)
    tokens = re.split(r"[^a-z0-9]+", norm)
    keywords = []
    seen: set = set()
    for t in tokens:
        if not t or t in seen:
            continue
        if re.match(r"^\d{4,}$", t):
            keywords.append(t)
            seen.add(t)
        elif len(t) > 4 and t not in _STOPS:
            keywords.append(t)
            seen.add(t)
    return keywords


def extract_reference_terms(filename: str, keywords: list) -> list:
    """
    Extract specific identifiers from the filename to verify inside BOE document content.

    These are more precise than general keywords — they are used to confirm that a BOE
    publication actually mentions our document, not just shares a topic.

    Examples:
      ISO 50001-2018.pdf         → ["50001"]
      UNE-EN_16247-1=2023.pdf    → ["16247"]
      Directiva_2023_1791...pdf  → ["2023/1791", "1791/2023"]
    """
    stem = Path(filename).stem
    norm = _normalize(stem)
    refs: list = []

    # Directive-format reference: YYYY/NNN  (e.g. "Directiva_2023_1791" → "2023/1791")
    for m in re.finditer(r"(\d{4})[/_\s-]+(\d{3,4})(?!\d)", norm):
        y, n = m.group(1), m.group(2)
        if 2000 <= int(y) <= 2030:
            refs.append(f"{y}/{n}")
            refs.append(f"{n}/{y}")

    # 5+ digit numbers that aren't standalone years (norm numbers like 50001, 16247)
    for n in re.findall(r"\d{5,}", norm):
        refs.append(n)

    # ISO/UNE specific: extract number after the standard prefix
    m = re.search(r"(?:iso|une)[_\s-]*(\d{4,})", norm)
    if m and m.group(1) not in refs:
        refs.append(m.group(1))

    # Known acronyms that are precise enough to anchor a content search
    tokens = re.split(r"[^a-z0-9]+", norm)
    for t in tokens:
        if t in _KNOWN_ACRONYMS and t not in refs:
            refs.append(t)

    seen: set = set()
    result = []
    for r in refs:
        rn = _normalize(r)
        if rn and rn not in seen:
            seen.add(rn)
            result.append(rn)
    return result


# ---------------------------------------------------------------------------
# Document type routing
# ---------------------------------------------------------------------------


def _classify_document(filename: str) -> str:
    """Route a document to its verification pipeline.

    Returns: 'UNE_ISO' | 'DIRECTIVA' | 'BOE_CANDIDATE'
    """
    if _TRACKER_AVAILABLE:
        return classify_document_origin(filename)

    # Fallback regex when normative_tracker not installed
    fn_norm = _normalize(filename)
    if re.search(r"(?:iso|une)[\s/_-]", fn_norm):
        return "UNE_ISO"
    if re.search(r"directiva", fn_norm):
        return "DIRECTIVA"
    return "BOE_CANDIDATE"


def _is_state_norm(item: dict) -> bool:
    """Return True if the BOE item is state-level (not regional/autonomous community)."""
    dept = _normalize(str(item.get("departamento", "")))
    return not bool(_REGIONAL_DEPT_RE.search(dept))


# ---------------------------------------------------------------------------
# UNE/ISO catalog pipeline
# ---------------------------------------------------------------------------

_UNE_ISO_CHANGE_MAP = {
    "SIN_CAMBIO": "en_vigor",
    "REVISION_EN_CURSO": "posible_cambio",
    "CORRIGENDUM": "posible_cambio",
    "ENMIENDA_PUBLICADA": "modificado_parcialmente",
    "REVISION_COMPLETA": "modificado_parcialmente",
    "SUSTITUCION": "sustituido",
    "ANULACION": "derogado_total",
}


def _check_une_iso(filename: str, entry: dict) -> dict:
    """Verify a UNE/ISO standard against AENOR/ISO public catalogs."""
    indexed_date = entry.get("indexed_date", date.today().isoformat())

    if not _TRACKER_AVAILABLE or not _UNE_ISO_CHECKER_AVAILABLE:
        return build_verification(
            filename,
            "no_encontrado",
            0.50,
            None,
            indexed_date,
            [],
            [],
            method="catalogo_une_iso_no_disponible",
        )

    identity = parse_une_iso_identifier(filename)
    if identity is None:
        return build_verification(
            filename,
            "no_encontrado",
            0.50,
            None,
            indexed_date,
            [],
            [],
            method="catalogo_une_iso",
        )

    try:
        candidate = UNEISOChecker(vector_store=None).check(identity)
    except Exception as exc:
        print(f"[Monitor]   UNEISOChecker error for {filename}: {exc}")
        return build_verification(
            filename,
            "error",
            0.0,
            None,
            indexed_date,
            [],
            [],
            method="catalogo_une_iso",
        )

    estado = _UNE_ISO_CHANGE_MAP.get(candidate.change_type, "posible_cambio")
    confianza = min(candidate.raw_score / 90.0, 1.0)

    evidencia = {
        "efecto_juridico": candidate.change_type,
        "zona_documento": "catalogo",
        "disposicion": candidate.catalog_source or "",
        "articulos_afectados": [],
        "texto_evidencia": (
            f"Identificador en catálogo: {candidate.catalog_identifier}. "
            f"Estado: {candidate.catalog_status}"
        )[:300],
        "norma_causante": candidate.catalog_identifier or "",
        "fecha_vigor": candidate.catalog_date or "",
        "norma_sucesora": candidate.successor_ref or "",
    }

    return build_verification(
        filename,
        estado,
        confianza,
        None,
        indexed_date,
        [],
        [],
        method="catalogo_une_iso",
        evidencia=evidencia,
    )


# ---------------------------------------------------------------------------
# BOE content fetching — section-aware
# ---------------------------------------------------------------------------


def _boe_id_from_url(url: str) -> str | None:
    """Extract BOE-A-YYYY-NNNNN identifier from a URL."""
    m = _BOE_ID_RE.search(url)
    return m.group(0).upper() if m else None


def fetch_boe_sections(url: str) -> dict:
    """
    Download a BOE publication and split its text into legal sections.

    Returns dict with keys:
      preambulo, articulos, disp_adicionales, disp_transitorias,
      disp_derogatorias, disp_finales, full_text
    """
    sections: dict = {
        k: ""
        for k in (
            "preambulo",
            "articulos",
            "disp_adicionales",
            "disp_transitorias",
            "disp_derogatorias",
            "disp_finales",
            "full_text",
        )
    }

    boe_id = _boe_id_from_url(url)
    raw_text = ""

    try:
        with httpx.Client(verify=True, follow_redirects=True) as client:
            if boe_id:
                xml_url = f"https://www.boe.es/diario_boe/xml.php?id={boe_id}"
                r = client.get(xml_url, timeout=8.0)
                if r.status_code == 200 and "<texto" in r.text:
                    raw_text = r.text

            if not raw_text:
                r = client.get(url, timeout=8.0)
                if r.status_code == 200:
                    raw_text = r.text
    except Exception as exc:
        print(f"[Monitor]   fetch_boe_sections error ({url[:60]}…): {exc}")
        return sections

    if not raw_text:
        return sections

    plain = re.sub(r"<[^>]+>", " ", raw_text)
    plain = re.sub(r"\s+", " ", plain)
    norm = _normalize(plain)

    sections["full_text"] = norm

    # Find positions of all section header occurrences
    markers: list = []
    for section_key, pattern in _SECTION_HEADERS.items():
        for m in pattern.finditer(norm):
            markers.append((m.start(), section_key))

    if not markers:
        # No section structure — assign first 500 chars as preámbulo (best effort)
        sections["preambulo"] = norm[:500]
        return sections

    markers.sort(key=lambda x: x[0])

    # Text before the first marker is the preámbulo
    sections["preambulo"] = norm[: markers[0][0]]

    for idx, (pos, key) in enumerate(markers):
        end = markers[idx + 1][0] if idx + 1 < len(markers) else len(norm)
        chunk = norm[pos:end]
        sections[key] = (
            (sections[key] + " " + chunk).strip() if sections[key] else chunk
        )

    return sections


# ---------------------------------------------------------------------------
# Legal effect analysis by section
# ---------------------------------------------------------------------------


def check_legal_effect(sections: dict, ref_terms: list) -> dict:
    """
    Determine the legal effect of a BOE publication on the indexed document.

    Priority (stops at first positive match):
    1. disp_derogatorias + ref_term + derogation term → derogado_total, conf 0.92
    2. disp_derogatorias + ref_term (no explicit derog term) → derogado_total, conf 0.75
    3. articulos/disp_finales/disp_adicionales + ref_term + modify term → modificado_parcialmente, conf 0.82
    4. preambulo/full_text + ref_term → referenciado_sin_efecto, conf 0.45
    5. No ref_term found → effect_type None
    """
    if not ref_terms:
        return {
            "effect_type": None,
            "zona": None,
            "articulos_afectados": [],
            "texto_evidencia": "",
            "confianza": 0.0,
        }

    def _find_ref(text: str) -> tuple[bool, str]:
        for term in ref_terms:
            idx = text.find(term)
            if idx >= 0:
                snippet = text[max(0, idx - 80) : idx + 120].strip()
                return True, snippet
        return False, ""

    def _arts(text: str) -> list:
        if _TRACKER_AVAILABLE:
            return extract_affected_articles(text)
        return list(
            dict.fromkeys(re.findall(r"art[ií]culo\s+(\d+(?:\.\d+)?)", text, re.I))
        )

    # 1 & 2: disposición derogatoria
    derog = sections.get("disp_derogatorias", "")
    if derog:
        found, snippet = _find_ref(derog)
        if found:
            conf = 0.92 if _DEROGATE.search(derog) else 0.75
            return {
                "effect_type": "derogado_total",
                "zona": "disp_derogatorias",
                "articulos_afectados": _arts(derog),
                "texto_evidencia": snippet[:300],
                "confianza": conf,
            }

    # 3: artículos / disposiciones finales / adicionales
    for zone_key in ("articulos", "disp_finales", "disp_adicionales"):
        zone = sections.get(zone_key, "")
        if not zone:
            continue
        found, snippet = _find_ref(zone)
        if found and _MODIFY.search(zone):
            return {
                "effect_type": "modificado_parcialmente",
                "zona": zone_key,
                "articulos_afectados": _arts(zone),
                "texto_evidencia": snippet[:300],
                "confianza": 0.82,
            }

    # 4: preámbulo / full_text (referencia sin efecto jurídico)
    for zone_key in ("preambulo", "full_text"):
        zone = sections.get(zone_key, "")
        if not zone:
            continue
        found, snippet = _find_ref(zone)
        if found:
            return {
                "effect_type": "referenciado_sin_efecto",
                "zona": zone_key,
                "articulos_afectados": [],
                "texto_evidencia": snippet[:300],
                "confianza": 0.45,
            }

    return {
        "effect_type": None,
        "zona": None,
        "articulos_afectados": [],
        "texto_evidencia": "",
        "confianza": 0.0,
    }


# ---------------------------------------------------------------------------
# BOE reference extraction from filename
# ---------------------------------------------------------------------------

_BOE_REF_RE = re.compile(r"BOE-[A-Z]-\d{4}-\d+", re.I)
_NORM_REF_RE = re.compile(
    r"(?:RD|Ley|Directiva|Reglamento)[_\s-]*(\d+)[/_\-](\d{4})", re.I
)


def extract_boe_ref(filename: str) -> str:
    stem = Path(filename).stem
    m = _BOE_REF_RE.search(stem)
    if m:
        return m.group(0).upper()
    m = _NORM_REF_RE.search(stem)
    if m:
        return m.group(0)
    return ""


# ---------------------------------------------------------------------------
# Status inference — BOE pipeline
# ---------------------------------------------------------------------------


def infer_status(
    items: list,
    indexed_date: str,
    keywords: list,
    ref_terms: list,
) -> tuple:
    """
    Infer document vigency status with section-aware legal analysis.

    Returns (status, confidence, best_item, effect_dict | None).

    Stage 1 — Jurisdiction + keyword title filter:
        Only state-level BOE items whose title contains at least one keyword.

    Stage 2 — Section-aware content verification:
        fetch_boe_sections() + check_legal_effect() per candidate.
    """
    if not items:
        return "no_encontrado", 0.70, None, None

    state_items = [i for i in items if _is_state_norm(i)]
    if not state_items:
        return "en_vigor", 0.70, items[0] if items else None, None

    newer = [
        i for i in state_items if i.get("fecha_actualizacion", "")[:10] > indexed_date
    ]

    if not newer:
        best = max(state_items, key=lambda i: i.get("fecha_actualizacion", ""))
        return "en_vigor", 0.85, best, None

    newer.sort(key=lambda i: i.get("fecha_actualizacion", ""), reverse=True)

    title_matches = [
        item
        for item in newer
        if any(kw in _normalize(item.get("titulo", "")) for kw in keywords)
    ]

    if not title_matches:
        return "en_vigor", 0.65, newer[0], None

    # Stage 2: section-aware content verification
    best_effect: dict | None = None
    best_item = None
    best_conf = 0.0

    for item in title_matches[:3]:
        url = item.get("url", "")
        if not url:
            continue
        time.sleep(0.35)
        sections = fetch_boe_sections(url)
        if not sections.get("full_text"):
            continue

        if ref_terms:
            effect = check_legal_effect(sections, ref_terms)
            if effect["effect_type"] is not None and effect["confianza"] > best_conf:
                best_effect = effect
                best_item = item
                best_conf = effect["confianza"]
                if effect["effect_type"] == "derogado_total" and best_conf >= 0.90:
                    break
        else:
            # No ref_terms: check title for derogation signal only
            titulo_norm = _normalize(item.get("titulo", ""))
            if _DEROGATE.search(titulo_norm):
                best_effect = {
                    "effect_type": "derogado_total",
                    "zona": "titulo",
                    "articulos_afectados": [],
                    "texto_evidencia": item.get("titulo", "")[:200],
                    "confianza": 0.65,
                }
                best_item = item
                best_conf = 0.65
                break

    if best_effect and best_effect["effect_type"]:
        return best_effect["effect_type"], best_conf, best_item, best_effect

    return "posible_cambio", 0.52, title_matches[0], None


# ---------------------------------------------------------------------------
# Build verification result dict
# ---------------------------------------------------------------------------


def build_verification(
    filename: str,
    status: str,
    confidence: float,
    best_item,
    indexed_date: str,
    keywords: list,
    ref_terms: list,
    method: str = "keyword",
    evidencia: dict | None = None,
) -> dict:
    boe_ref = extract_boe_ref(filename)
    if not boe_ref and best_item:
        boe_ref = best_item.get("id", "") or best_item.get("identificador", "")

    result = {
        "documento": filename,
        "referencia_boe": boe_ref or None,
        "fecha_publicacion": indexed_date,
        "estado": status,
        "nivel_alerta": _NIVEL_ALERTA.get(status, "gris"),
        "ultima_actualizacion_detectada": (
            best_item["fecha_actualizacion"][:10] if best_item else None
        ),
        "version_vigente_url": best_item.get("url") if best_item else None,
        "resultado_verificacion": _RESULTADO_TEXT.get(status, ""),
        "recomendacion": _RECOMENDACION_TEXT.get(status, ""),
        "confianza": round(confidence, 2),
        "trazabilidad": {
            "fecha_consulta": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fuente": "BOE — Legislación Consolidada",
            "metodo": method,
            "terminos_referencia": ref_terms,
            "accion_recomendada": _ACCION.get(status, "reintentar"),
            "accion_aplicada": None,
        },
    }

    if evidencia:
        result["evidencia"] = evidencia

    return result


def _zona_label(zona: str) -> str:
    return {
        "disp_derogatorias": "Disposición Derogatoria",
        "disp_finales": "Disposición Final",
        "disp_adicionales": "Disposición Adicional",
        "disp_transitorias": "Disposición Transitoria",
        "articulos": "Articulado",
        "preambulo": "Preámbulo / Exposición de Motivos",
        "full_text": "Texto completo (sin estructura identificada)",
        "titulo": "Título del documento",
        "catalogo": "Catálogo AENOR/ISO",
    }.get(zona, zona)


# ---------------------------------------------------------------------------
# Alert state persistence (atomic writes)
# ---------------------------------------------------------------------------


def load_alerts() -> dict:
    if ALERTS_FILE.exists():
        try:
            return json.loads(ALERTS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Monitor] Warning: could not read alerts file: {e}")
    return {}


def save_alerts(state: dict) -> None:
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ALERTS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ALERTS_FILE)


def _write_status(status: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATUS_FILE)


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------


def run_check() -> None:
    """
    For each indexed PDF:
    1. Classify document type (UNE_ISO / DIRECTIVA / BOE_CANDIDATE).
    2. Route to appropriate pipeline:
       - UNE_ISO → _check_une_iso() via AENOR/ISO catalogs
       - BOE_CANDIDATE / DIRECTIVA → BOE API with section-aware legal analysis
    3. Write structured verification result with evidencia field.
    """
    print(f"[Monitor] BOE check started — {datetime.now().isoformat()}")
    print(
        f"[Monitor] normative_tracker: {'disponible' if _TRACKER_AVAILABLE else 'no disponible (fallback activo)'}"
    )

    loader = PDFLoader()
    pdfs = loader.get_pdf_list()
    started_at = datetime.now().isoformat()

    if not pdfs:
        print("[Monitor] No PDFs indexed. Nothing to check.")
        _write_status(
            {
                "running": False,
                "started_at": started_at,
                "completed_at": datetime.now().isoformat(),
                "total": 0,
                "current": 0,
                "current_doc": None,
                "phase": "Sin documentos indexados",
                "error": None,
                "docs_done": [],
            }
        )
        return

    alerts_state = load_alerts()
    today = date.today().isoformat()
    client = BOESearchClient()
    total = len(pdfs)
    docs_done: list = []

    _write_status(
        {
            "running": True,
            "started_at": started_at,
            "completed_at": None,
            "total": total,
            "current": 0,
            "current_doc": None,
            "phase": "Iniciando verificación",
            "error": None,
            "docs_done": [],
        }
    )

    for idx, filename in enumerate(pdfs):
        print(f"[Monitor] Checking ({idx + 1}/{total}): {filename}")

        def _progress(phase: str) -> None:
            _write_status(
                {
                    "running": True,
                    "started_at": started_at,
                    "completed_at": None,
                    "total": total,
                    "current": idx + 1,
                    "current_doc": filename,
                    "phase": phase,
                    "error": None,
                    "docs_done": list(docs_done),
                }
            )

        _progress("Clasificando documento")

        if filename not in alerts_state:
            alerts_state[filename] = {
                "indexed_date": today,
                "keywords": [],
                "ref_terms": [],
                "alerts": [],
                "last_check": None,
            }

        entry = alerts_state[filename]
        kws = extract_keywords(filename)
        ref_terms = extract_reference_terms(filename, kws)
        entry["keywords"] = kws
        entry["ref_terms"] = ref_terms
        indexed_date = entry.get("indexed_date", today)

        origin = _classify_document(filename)
        print(f"[Monitor]   origin={origin}  keywords={kws}  ref_terms={ref_terms}")

        # UNE/ISO → catalog pipeline (never BOE)
        if origin == "UNE_ISO":
            _progress("Consultando catálogo AENOR/ISO")
            try:
                entry["verification"] = _check_une_iso(filename, entry)
            except Exception as e:
                print(f"[Monitor]   UNE/ISO check error for '{filename}': {e}")
                entry["verification"] = build_verification(
                    filename, "error", 0.0, None, indexed_date, [], [], "error"
                )
            entry["last_check"] = datetime.now().isoformat()
            docs_done.append(filename)
            _progress("Resultado obtenido")
            continue

        # BOE pipeline (BOE_CANDIDATE or DIRECTIVA)
        if not kws:
            print(f"[Monitor]   No keywords for {filename}, skipping.")
            entry["last_check"] = datetime.now().isoformat()
            entry["verification"] = build_verification(
                filename,
                "no_encontrado",
                0.70,
                None,
                indexed_date,
                [],
                [],
                "sin_keywords",
            )
            docs_done.append(filename)
            _progress("Resultado obtenido")
            continue

        _progress("Consultando BOE")
        all_items: list = []
        existing_urls = {a["url"] for a in entry.get("alerts", []) if a.get("url")}
        new_alerts: list = []

        for kw in kws:
            try:
                raw = client.search_regulations(texto=kw, limit=5)
            except Exception as e:
                print(f"[Monitor]   Search error for '{kw}': {e}")
                continue
            items = BOESearchClient.extract_items(raw)
            state_items = [i for i in items if _is_state_norm(i)]
            all_items.extend(state_items)

            for item in state_items:
                fecha = item.get("fecha_actualizacion", "")
                url = item.get("url", "")
                titulo = item.get("titulo", "")
                titulo_norm = _normalize(titulo)

                if (
                    fecha
                    and fecha[:10] > indexed_date
                    and url
                    and url not in existing_urls
                    and kw in titulo_norm
                ):
                    new_alerts.append(
                        {
                            "date": fecha[:10],
                            "title": titulo,
                            "url": url,
                            "rango": item.get("rango", ""),
                            "departamento": item.get("departamento", ""),
                            "matched_keyword": kw,
                        }
                    )
                    existing_urls.add(url)

        _progress("Verificando contenido")

        try:
            method = "secciones_juridicas" if ref_terms else "titulo"
            status, confidence, best_item, effect = infer_status(
                all_items, indexed_date, kws, ref_terms
            )
        except Exception as e:
            print(f"[Monitor]   Status inference error for '{filename}': {e}")
            status, confidence, best_item, effect = "error", 0.0, None, None
            method = "error"

        evidencia = None
        if effect and effect.get("effect_type"):
            evidencia = {
                "efecto_juridico": effect["effect_type"],
                "zona_documento": effect.get("zona", ""),
                "disposicion": _zona_label(effect.get("zona", "")),
                "articulos_afectados": effect.get("articulos_afectados", []),
                "texto_evidencia": effect.get("texto_evidencia", "")[:300],
                "norma_causante": (best_item.get("titulo", "") if best_item else "")[
                    :120
                ],
                "fecha_vigor": (
                    best_item.get("fecha_actualizacion", "") if best_item else ""
                )[:10],
            }

        entry["verification"] = build_verification(
            filename,
            status,
            confidence,
            best_item,
            indexed_date,
            kws,
            ref_terms,
            method,
            evidencia,
        )

        if new_alerts:
            print(
                f"[Monitor]   {len(new_alerts)} new alert(s) for {filename} — status: {status}"
            )
            entry["alerts"].extend(new_alerts)
        else:
            print(f"[Monitor]   No new alerts for {filename} — status: {status}")

        entry["last_check"] = datetime.now().isoformat()
        docs_done.append(filename)
        _progress("Resultado obtenido")

    save_alerts(alerts_state)
    total_alerts = sum(len(v.get("alerts", [])) for v in alerts_state.values())
    print(
        f"[Monitor] Check complete — {total_alerts} total alert(s) across all documents."
    )

    _write_status(
        {
            "running": False,
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(),
            "total": total,
            "current": total,
            "current_doc": None,
            "phase": "Verificación completada",
            "error": None,
            "docs_done": list(docs_done),
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_check()

# ---------------------------------------------------------------------------
# Windows Task Scheduler setup (run once as admin in PowerShell)
# ---------------------------------------------------------------------------
# schtasks /Create /TN "Tambora\BOEMonitor" ^
#   /TR "\"C:\Users\CesarSuelaCedenilla\Desktop\JARVIS\Tambora_servidor\.venv\Scripts\python.exe\" \"C:\Users\CesarSuelaCedenilla\Desktop\JARVIS\Tambora_servidor\boe_monitor.py\"" ^
#   /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 10:00 /F
