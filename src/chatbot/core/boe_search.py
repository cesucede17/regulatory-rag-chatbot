"""BOE legislación consolidada search client.

The BOE open-data API for legislación consolidada
(https://www.boe.es/datosabiertos/api/legislacion-consolidada) returns
a paginated list of recently-updated legislation ordered by update date.
It does NOT support text filtering — any extra query parameter returns 400.

Strategy: fetch pages and filter in-memory using normalized keyword matching.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings

# Spanish stopwords to strip before keyword matching
_STOPWORDS = {
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "de",
    "del",
    "en",
    "a",
    "al",
    "por",
    "para",
    "con",
    "sin",
    "que",
    "se",
    "su",
    "sus",
    "es",
    "son",
    "fue",
    "han",
    "hay",
    "como",
    "pero",
    "mas",
    "si",
    "no",
    "ya",
    "sobre",
    "entre",
    "desde",
    "hasta",
    "ante",
    "bajo",
    "que",
    "cual",
    "cuales",
    "como",
    "donde",
    "cuando",
    # Command verbs often present in user queries
    "busca",
    "buscar",
    "dame",
    "dime",
    "muestra",
    "lista",
    "encuentra",
    "encontrar",
    "quiero",
    "necesito",
    "hay",
    "existe",
    "puedes",
    "informacion",
    "acerca",
}


def _normalize(text: str) -> str:
    """Lowercase + remove accents + ASCII only."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return nfkd.encode("ascii", "ignore").decode("ascii")


def _extract_keywords(query: str) -> List[str]:
    """Extract significant keywords from a free-form query."""
    norm = _normalize(query)
    # Remove punctuation
    norm = re.sub(r"[¿?¡!()\[\]{}/\\\"']", " ", norm)
    words = norm.split()
    # Keep words that are not stopwords and have length > 2
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


# Synonyms for specialized terms that may not appear verbatim in BOE titles
_SYNONYMS: dict[str, list[str]] = {
    "agregador": ["agregador", "gestor", "comercializador", "demanda"],
    "autoconsumo": ["autoconsumo", "autoproductor", "generacion distribuida"],
    "almacenamiento": ["almacenamiento", "bateria", "acumulacion"],
    "renovable": [
        "renovable",
        "solar",
        "eolico",
        "fotovoltaico",
        "hidraulica",
        "biomasa",
        "biogas",
    ],
    "mercado": ["mercado", "operador", "pool", "omie", "esios"],
    "red": ["red", "transporte", "distribucion", "acceso"],
    "balance": ["balance", "neto", "compensacion"],
    "ppa": ["ppa", "contrato", "suministro"],
    "tarifa": ["tarifa", "precio", "peaje", "cargo"],
    "eficiencia": ["eficiencia", "ahorro", "consumo", "certificado"],
    # Biomasa y bioenergía
    "biomasa": [
        "biomasa",
        "biocombustible",
        "biocarburante",
        "biogas",
        "biogás",
        "bioenergia",
        "bioenergía",
        "residuos organicos",
        "cogeneracion",
        "biomass",
        "bioliquido",
        "bioliquidos",
    ],
    "biogas": ["biogas", "biogás", "biomasa", "residuos organicos", "cogeneracion"],
    "biocombustible": ["biocombustible", "biocarburante", "biomasa", "bioliquido"],
    # Energía y electricidad
    "energia": [
        "energia",
        "electrica",
        "electricidad",
        "potencia",
        "generacion electrica",
    ],
    "cogeneracion": [
        "cogeneracion",
        "cogeneración",
        "residuos",
        "biomasa",
        "eficiencia energetica",
    ],
    # Medio ambiente y residuos
    "residuos": [
        "residuos",
        "vertedero",
        "reciclaje",
        "gestion residuos",
        "biorresiduos",
    ],
    "medioambiente": [
        "medioambiente",
        "ambiental",
        "contaminacion",
        "emisiones",
        "co2",
    ],
    "agua": ["agua", "hidraulica", "hidrico", "cuenca", "riego", "saneamiento"],
    # Sector agrícola y forestal
    "forestal": ["forestal", "bosque", "monte", "madera", "biomasa", "incendio"],
    "agricultura": ["agricola", "agricultura", "agroalimentario", "ganadero", "rural"],
    # Contratación y licitación
    "licitacion": [
        "licitacion",
        "concurso",
        "adjudicacion",
        "contrato publico",
        "contratacion",
    ],
    # Tributos
    "impuesto": ["impuesto", "tributo", "fiscal", "hacienda", "irpf", "iva", "tasas"],
    # Laboral
    "laboral": [
        "laboral",
        "trabajo",
        "empleo",
        "convenio",
        "seguridad social",
        "salario",
    ],
    # Seguridad industrial
    "industria": [
        "industria",
        "industrial",
        "seguridad industrial",
        "normativa tecnica",
    ],
}

# Pattern for specific norm references: "RD 244/2019", "RDL 7/2026", "Ley 24/2013"
# Group 1 = type prefix (e.g. "rd", "Real Decreto"); Group 2 = number (e.g. "36/2023")
_NORM_REF_RE = re.compile(
    r"(real\s+decreto(?:-ley)?|rd(?:l)?|ley(?:\s+organica)?|lo|orden)\s*"
    r"(\d+/\d{4})",
    re.IGNORECASE,
)

# Map abbreviated/variant prefixes (after _normalize) to their canonical BOE form
_NORM_PREFIX_MAP: dict[str, str] = {
    "rd": "real decreto",
    "rdl": "real decreto-ley",
    "lo": "ley organica",
    "real decreto": "real decreto",
    "real decreto-ley": "real decreto-ley",
    "real decreto ley": "real decreto-ley",
    "ley organica": "ley organica",
    "ley": "ley",
    "orden": "orden",
}


def build_full_ref(prefix_raw: str, number: str) -> str:
    """Return the full normalised reference, e.g. 'real decreto 36/2023'.

    Used to match against BOE titles precisely, avoiding false positives like
    'Orden TMA/36/2023' when the user asked for 'Real Decreto 36/2023'.
    """
    prefix_norm = re.sub(r"\s+", " ", _normalize(prefix_raw).strip())
    full_prefix = _NORM_PREFIX_MAP.get(prefix_norm, prefix_norm)
    return f"{full_prefix} {_normalize(number)}"


def _expand_keywords(keywords: list[str]) -> list[str]:
    """Expand keywords with domain synonyms."""
    expanded = list(keywords)
    for kw in keywords:
        for base, syns in _SYNONYMS.items():
            if kw == base or kw in syns:
                for s in syns:
                    if s not in expanded:
                        expanded.append(s)
    return expanded


class BOESearchClient:
    """Client for BOE legislación consolidada API with in-memory keyword filtering."""

    _PAGE_SIZE = 50  # Max items per API call
    _MAX_PAGES = 20  # Max pages to scan (~1000 items)

    def __init__(self) -> None:
        self.base_url = settings.boe_legislacion_api_url

    @staticmethod
    def sanitize_texto(text: str) -> str:
        """Normalize query text (kept for backward-compat, used by workflow fallback)."""
        return _normalize(text)

    def search_regulations(
        self,
        texto: Optional[str] = None,
        materia: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Return legislation items matching the given keywords.

        Fetches pages from the BOE API and filters in-memory.
        Returns a dict compatible with extract_items().
        """
        # Build keyword list from available search terms
        combined = " ".join(filter(None, [texto, materia]))
        if not combined.strip():
            return None

        keywords = _extract_keywords(combined)
        if not keywords:
            return None

        # Expand with domain synonyms
        keywords_expanded = _expand_keywords(keywords)

        matching: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        page = 0
        scan_offset = offset

        headers = {"Accept": "application/json"}

        with httpx.Client(verify=True) as client:
            while len(matching) < limit and page < self._MAX_PAGES:
                params = {"offset": scan_offset, "limit": self._PAGE_SIZE}
                try:
                    response = client.get(
                        self.base_url,
                        params=params,
                        headers=headers,
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    data = response.json().get("data", [])
                except httpx.HTTPStatusError as exc:
                    print(f"BOE search HTTP {exc.response.status_code}: {exc}")
                    break
                except Exception as exc:
                    print(f"BOE search error: {exc}")
                    break

                if not isinstance(data, list) or not data:
                    break

                for item in data:
                    if not isinstance(item, dict):
                        continue
                    item_id = str(item.get("identificador", item.get("id", "")))
                    if item_id in seen_ids:
                        continue
                    titulo = _normalize(item.get("titulo", ""))
                    rango_raw = item.get("rango", "")
                    rango = _normalize(
                        rango_raw.get("texto", "")
                        if isinstance(rango_raw, dict)
                        else str(rango_raw)
                    )
                    departamento_raw = item.get("departamento", {})
                    departamento = _normalize(
                        departamento_raw.get("texto", "")
                        if isinstance(departamento_raw, dict)
                        else str(departamento_raw)
                    )
                    searchable = f"{titulo} {rango} {departamento}"
                    hay_match = any(kw in searchable for kw in keywords_expanded)
                    if hay_match:
                        matching.append(item)
                        seen_ids.add(item_id)
                    if len(matching) >= limit:
                        break

                scan_offset += self._PAGE_SIZE
                page += 1

        if not matching:
            return None

        return {"data": matching[:limit]}

    def search_by_norm_ref(self, norm_number: str) -> Optional[Dict[str, Any]]:
        """Search for a specific norm by its reference number (e.g. '244/2019', '7/2026')."""
        headers = {"Accept": "application/json"}
        matching: List[Dict[str, Any]] = []
        norm_norm = _normalize(norm_number)

        with httpx.Client(verify=True) as client:
            for page in range(self._MAX_PAGES):
                params = {"offset": page * self._PAGE_SIZE, "limit": self._PAGE_SIZE}
                try:
                    r = client.get(
                        self.base_url, params=params, headers=headers, timeout=30.0
                    )
                    r.raise_for_status()
                    data = r.json().get("data", [])
                except Exception as exc:
                    print(f"BOE norm ref search error: {exc}")
                    break
                if not isinstance(data, list) or not data:
                    break
                for item in data:
                    titulo = _normalize(item.get("titulo", ""))
                    if norm_norm in titulo:
                        matching.append(item)
                if matching:
                    break  # Found on this page, stop

        return {"data": matching} if matching else None

    def get_recent(self, limit: int = 20) -> Optional[Dict[str, Any]]:
        """Return the most recently updated legislation items (no filtering)."""
        headers = {"Accept": "application/json"}
        try:
            with httpx.Client(verify=True) as client:
                r = client.get(
                    self.base_url,
                    params={"offset": 0, "limit": limit},
                    headers=headers,
                    timeout=30.0,
                )
                r.raise_for_status()
                return r.json()
        except Exception as exc:
            print(f"BOE recent error: {exc}")
            return None

    @staticmethod
    def extract_items(raw: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract a flat list of legislation items from the API response.

        Each item has: id, titulo, url, fecha_actualizacion, rango, departamento.
        """
        if not raw:
            return []

        data = raw.get("data", [])
        if not isinstance(data, list):
            # Handle the old dict-wrapped format just in case
            if isinstance(data, dict):
                data = (
                    data.get("legislacion")
                    or data.get("response")
                    or data.get("items")
                    or []
                )

        if not data:
            return []

        results: List[Dict[str, str]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            entry: Dict[str, str] = {
                "id": str(item.get("identificador", item.get("id", ""))),
                "titulo": str(item.get("titulo", "Sin título")),
            }
            url = (
                item.get("url_html_consolidada")
                or item.get("url_eli")
                or item.get("url")
                or ""
            )
            if url:
                entry["url"] = str(url)
            fecha = (
                item.get("fecha_actualizacion") or item.get("fecha_disposicion") or ""
            )
            if fecha:
                entry["fecha_actualizacion"] = str(fecha)
            rango_raw = item.get("rango", "")
            if isinstance(rango_raw, dict):
                rango_raw = rango_raw.get("texto", "")
            if rango_raw:
                entry["rango"] = str(rango_raw)
            dept_raw = item.get("departamento", "")
            if isinstance(dept_raw, dict):
                dept_raw = dept_raw.get("texto", "")
            if dept_raw:
                entry["departamento"] = str(dept_raw)
            results.append(entry)

        return results
