"""Guarda mecánica (AST, no regex) de que este proyecto es solo el Chatbot BOE.

Sustituye al `tests/test_isolation.py` de cuando las dos herramientas vivían en
el mismo repositorio: entonces había que comprobar que dos paquetes vecinos no
se importaran entre sí. Ahora son proyectos separados, y lo que hay que
vigilar es lo contrario — que no vuelva a entrar código de auditorías por la
puerta de atrás (un fichero copiado, un import pegado), que aquí no resolvería
ni tendría sus dependencias declaradas.

Se recorre el AST (nodos `Import`/`ImportFrom`) en vez de hacer `grep`: la
palabra "auditorias" dentro de un docstring, un comentario o una cadena no es
una dependencia real, y un AST es la única forma de distinguirlo.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKAGES = (ROOT / "src" / "chatbot", ROOT / "src" / "shared")

# Nombres que no deben aparecer nunca: la otra herramienta, y los módulos de la
# estructura anterior (un repo único con `apps/` y un `src/` antes de eso).
FORBIDDEN = ("auditorias", "apps", "app_fastapi", "config", "auth", "common", "src")


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _absolute_imports(tree: ast.Module) -> set[str]:
    """Nombres de módulo de los imports ABSOLUTOS del árbol, a cualquier nivel
    de anidamiento (incluye los que están dentro de funciones). Los relativos
    (`from .core import x`) no pueden salir del paquete y se ignoran."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and not node.level:
                names.add(node.module)
    return names


def _violations(package_dir: Path, forbidden: tuple[str, ...]) -> list[str]:
    """El chequeo es por punto, no por `startswith` desnudo: `configparser` no
    es un submódulo de `config`, así que no debe marcarse."""
    out: list[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in sorted(_absolute_imports(tree)):
            for bad in forbidden:
                if module == bad or module.startswith(bad + "."):
                    out.append(f"{_display(path)}: import {module}")
    return out


def test_no_import_of_auditorias_or_the_old_layout():
    found = [v for pkg in PACKAGES for v in _violations(pkg, FORBIDDEN)]
    assert found == [], (
        "este proyecto solo es el Chatbot BOE; no debe importar nada de las "
        "auditorías ni de la estructura antigua:\n" + "\n".join(found)
    )


def test_project_has_no_auditorias_files():
    """Ni siquiera ficheros sueltos: si alguien copia aquí el paquete de
    auditorías, este test lo señala antes de que alguien lo importe."""
    stray = [str(p.relative_to(ROOT)) for p in ROOT.glob("auditorias*")]
    assert stray == [], f"restos de la otra herramienta en el proyecto: {stray}"


def test_detector_actually_flags_a_synthetic_violation(tmp_path):
    """Prueba negativa del detector: sin esto, los tests de arriba podrían
    estar pasando por una función rota que nunca encuentra nada."""
    (tmp_path / "malo.py").write_text(
        "import auditorias\nfrom auditorias.core import repository\n", encoding="utf-8"
    )
    (tmp_path / "bueno.py").write_text(
        "import os\nfrom chatbot.core import workflow\n", encoding="utf-8"
    )
    found = _violations(tmp_path, ("auditorias",))
    assert len(found) == 2
    assert all("malo.py" in v for v in found)


def test_detector_ignores_mentions_in_strings_and_comments(tmp_path):
    """Confirma que se usa AST y no una regex sobre el texto."""
    (tmp_path / "mencion.py").write_text(
        '"""Este proyecto no depende de auditorias."""\n'
        "# import auditorias -- comentario, no una importacion real\n"
        "x = 'auditorias'\n",
        encoding="utf-8",
    )
    assert _violations(tmp_path, ("auditorias",)) == []
