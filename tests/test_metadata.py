"""The citation metadata is a claim about identity, and it was wrong four ways.

`CITATION.cff` carried `doi: 10.5281/zenodo.21077401`, which resolves to *The Fragility of
Optimal-Agent Training* - the v1 preprint this paper audits and partly retracts. So the file told
GitHub, and anyone clicking "cite this repository", to cite the superseded result as if it were
version 2.0.1. The same file said `type: article` and `year: 2026`, neither of which the CFF 1.2.0
schema allows at the root (`type` is `software` or `dataset`; the date is `date-released`), and the
root has `additionalProperties: false`, so the whole file was invalid and the widget had nothing to
render. `pyproject.toml` still said 2.0.0 while the tree was tagged v2.0.1. And the README advertised
a 29-page PDF next to a 31-page one. The contact address was spelled two ways across five files -
`eliezer` in `pyproject.toml`, `elizer` in the citation file, the README and both paper sources - so
the test compares them to each other rather than to a typed literal. The preprint has two real DOIs,
a version one that pins the deposit and a concept one for all versions of that record; both are
guarded, because either could be mistaken for this work's identifier.

None of that is caught by a test that only asks whether the files exist. These ask whether they say
the same thing as the artifacts they point at. The page count is read out of the committed PDF rather
than trusted, the two version strings are compared to each other, and the preprint DOI is allowed to
stay - it is useful - but only where it is attributed to the preprint.

Run with pytest, or directly: python tests/test_metadata.py
"""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
CFF = REPO / "CITATION.cff"
README = REPO / "README.md"
PYPROJECT = REPO / "pyproject.toml"
PDF = REPO / "paper" / "paper_v2.pdf"

V1_DOIS = ("10.5281/zenodo.21077402", "10.5281/zenodo.21077401")
V1_TITLE = "The Fragility of Optimal-Agent Training"
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# the root properties of CFF 1.2.0, the version this file declares; the root is closed, so an
# invented key invalidates the whole document rather than being ignored
CFF_1_2_0_ROOT = {
    "abstract", "authors", "cff-version", "commit", "contact", "date-released", "doi",
    "identifiers", "keywords", "license", "license-url", "message", "preferred-citation",
    "references", "repository", "repository-artifact", "repository-code", "title", "type",
    "url", "version",
}
TOP_KEY = re.compile(r"^([a-z][a-z0-9-]*):", re.M)


def root_keys(text):
    return set(TOP_KEY.findall(text))


def test_the_citation_file_declares_a_schema_version_this_test_knows():
    text = CFF.read_text(encoding="utf-8")
    assert re.search(r"^cff-version:\s*1\.2\.0\s*$", text, re.M), (
        "the root key list below is the 1.2.0 one; a newer CFF needs a newer list")


def test_the_citation_root_uses_only_keys_the_schema_allows():
    text = CFF.read_text(encoding="utf-8")
    invented = root_keys(text) - CFF_1_2_0_ROOT
    assert not invented, (
        "CFF 1.2.0 closes the root, so these keys invalidate the file: %s" % ", ".join(sorted(invented)))
    kind = re.search(r"^type:\s*(\S+)", text, re.M)
    assert kind and kind.group(1) in ("software", "dataset"), (
        "the root `type` is software or dataset; an article goes under preferred-citation or references")


def test_the_preprint_doi_is_not_offered_as_this_works_doi():
    text = CFF.read_text(encoding="utf-8")
    top_doi = re.search(r"^doi:\s*(.+)$", text, re.M)
    for doi in V1_DOIS:
        assert not (top_doi and doi in top_doi.group(1)), (
            "the root DOI is what a citer gets, and %s is the preprint this version retracts" % doi)
        if doi in text:
            assert V1_TITLE in text, "the preprint DOI is present but nothing says which work it is"
            for line in text.splitlines():
                if doi in line:
                    assert line[:1].isspace(), (
                        "the preprint DOI must sit inside a reference, not at the root: %r" % line)


def test_the_readme_attributes_the_preprint_doi_to_the_preprint():
    text = README.read_text(encoding="utf-8")
    for para in re.split(r"\n\s*\n", text):
        if any(doi in para for doi in V1_DOIS):
            low = para.lower()
            assert "v1" in low and "preprint" in low, (
                "the README links a preprint DOI without saying it is the earlier v1 preprint:\n%s" % para)


def test_the_author_email_is_spelled_the_same_everywhere():
    carriers = (CFF, README, PYPROJECT, REPO / "paper" / "paper.tex", REPO / "paper" / "paper_v2.tex")
    seen = {}
    for path in carriers:
        found = set(EMAIL.findall(path.read_text(encoding="utf-8")))
        assert found, "%s carries no email, so this test cannot compare it" % path.name
        seen[path.name] = found
    distinct = set().union(*seen.values())
    assert len(distinct) == 1, (
        "the contact address disagrees across the metadata: %s"
        % "; ".join("%s=%s" % (k, ",".join(sorted(v))) for k, v in sorted(seen.items())))


def test_the_readme_page_count_is_the_pdfs_page_count():
    from pypdf import PdfReader

    text = README.read_text(encoding="utf-8")
    claimed = re.search(r"Paper \(PDF, (\d+) pages?\)", text)
    assert claimed, "the README no longer states a page count; this test is vacuous"
    actual = len(PdfReader(str(PDF)).pages)
    assert int(claimed.group(1)) == actual, (
        "the README says %s pages and paper/paper_v2.pdf has %d" % (claimed.group(1), actual))


def test_the_package_and_the_citation_agree_on_the_version():
    cff = re.search(r'^version:\s*"?([0-9][^"\s]*)"?', CFF.read_text(encoding="utf-8"), re.M)
    pkg = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.M)
    assert cff and pkg, "one of the two files stopped declaring a version"
    assert cff.group(1) == pkg.group(1), (
        "CITATION.cff says %s and pyproject.toml says %s, so a wheel and a citation disagree"
        % (pkg.group(1), cff.group(1)))


if __name__ == "__main__":
    for fn in (test_the_citation_file_declares_a_schema_version_this_test_knows,
               test_the_citation_root_uses_only_keys_the_schema_allows,
               test_the_preprint_doi_is_not_offered_as_this_works_doi,
               test_the_readme_attributes_the_preprint_doi_to_the_preprint,
               test_the_readme_page_count_is_the_pdfs_page_count,
               test_the_package_and_the_citation_agree_on_the_version,
               test_the_author_email_is_spelled_the_same_everywhere):
        fn()
        print("ok %s" % fn.__name__)