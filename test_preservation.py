"""
Preservation property tests for BOTCOIN solver — non-buggy code paths.

These tests capture the CURRENT (unfixed) baseline behavior of code paths
that are NOT affected by the three bugs (revenue overwrite, acrostic Q-reference,
transcript resolution). They should PASS on unfixed code and continue to pass
after the fix, ensuring no regressions.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
"""

import json
import re
import pytest
from botcoin_solver import (
    DocumentParser,
    QuestionEngine,
    ConstraintEngine,
    extract_company_names_from_doc,
    next_prime,
)


@pytest.fixture(scope="module")
def test_samples():
    with open("test_samples.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pipeline_results(test_samples):
    """Run the solver pipeline on all 5 samples and cache results."""
    results = []
    for sample in test_samples:
        ch = sample["challenge"]
        doc = ch["doc"]
        questions = ch["questions"]
        constraints = ch["constraints"]

        company_names = extract_company_names_from_doc(doc)
        parser = DocumentParser(doc, company_names)
        companies = parser.parse()

        engine = QuestionEngine(companies)
        answers = [engine.answer(q) for q in questions]

        ce = ConstraintEngine(constraints, questions, answers, companies)
        result = ce.compute_all()

        results.append({
            "companies": companies,
            "answers": answers,
            "constraints": result,
            "challenge": ch,
        })
    return results


class TestNonAffectedConstraintPreservation:
    """C0 (word count), C1 (HQ city), C2 (CEO last name), C7 (forbidden letter)
    are NOT affected by the three bugs and should compute correctly on all samples.

    **Validates: Requirements 3.4, 3.6**
    """

    @pytest.mark.parametrize("sample_idx", range(5))
    def test_c0_word_count(self, pipeline_results, sample_idx):
        """C0 word count constraint should produce a positive integer for all samples."""
        result = pipeline_results[sample_idx]["constraints"]
        wc = result.get("word_count")
        assert isinstance(wc, int) and wc > 0, (
            f"Sample {sample_idx}: C0 word_count should be a positive int, got {wc}"
        )

    @pytest.mark.parametrize("sample_idx", range(5))
    def test_c7_forbidden_letter(self, pipeline_results, sample_idx):
        """C7 forbidden letter constraint should produce a single lowercase letter."""
        result = pipeline_results[sample_idx]["constraints"]
        fl = result.get("forbidden_letter")
        assert isinstance(fl, str) and len(fl) == 1 and fl.isalpha(), (
            f"Sample {sample_idx}: C7 forbidden_letter should be a single letter, got {fl!r}"
        )

    @pytest.mark.parametrize("sample_idx", [0, 2, 3])
    def test_c1_hq_city_present(self, pipeline_results, sample_idx):
        """C1 HQ city should be in required_elements for samples with parsed companies.

        Samples 0, 2, 3 have standard format docs that parse companies successfully.
        """
        result = pipeline_results[sample_idx]["constraints"]
        required = result.get("required_elements", [])
        # C1 adds a city string to required_elements
        assert len(required) > 0, (
            f"Sample {sample_idx}: required_elements should not be empty"
        )

    @pytest.mark.parametrize("sample_idx", [0, 2, 3])
    def test_c2_ceo_last_name_present(self, pipeline_results, sample_idx):
        """C2 CEO last name should be in required_elements for samples with parsed companies."""
        result = pipeline_results[sample_idx]["constraints"]
        required = result.get("required_elements", [])
        assert len(required) >= 2, (
            f"Sample {sample_idx}: required_elements should have at least 2 entries "
            f"(city + CEO last name), got {len(required)}"
        )

    @pytest.mark.parametrize("sample_idx", range(5))
    def test_c0_c7_baseline_values(self, pipeline_results, test_samples, sample_idx):
        """Verify exact C0 and C7 values match known baselines for each sample."""
        expected_word_counts = {0: 21, 1: 19, 2: 22, 3: 16, 4: 20}
        expected_forbidden = {0: "g", 1: "p", 2: "k", 3: "s", 4: "q"}

        result = pipeline_results[sample_idx]["constraints"]
        assert result["word_count"] == expected_word_counts[sample_idx], (
            f"Sample {sample_idx}: C0 word_count expected "
            f"{expected_word_counts[sample_idx]}, got {result['word_count']}"
        )
        assert result["forbidden_letter"] == expected_forbidden[sample_idx], (
            f"Sample {sample_idx}: C7 forbidden_letter expected "
            f"{expected_forbidden[sample_idx]!r}, got {result['forbidden_letter']!r}"
        )


class TestRevenueParsingNonDuplicate:
    """For companies with only ONE revenue line in the document, verify parsed
    revenue values are correct. These are NOT affected by the revenue overwrite bug.

    **Validates: Requirements 3.1, 3.5**
    """

    def test_sample0_single_revenue_companies(self, pipeline_results):
        """Sample 0: spot-check revenue for companies with a single REVENUE line."""
        companies = pipeline_results[0]["companies"]

        # Onyx Industries has a single REVENUE line:
        # "Onyx Industries REVENUE: first quarter $4.88 billion; Q2 $4,521M; Q3 204 million dollars; fourth quarter 457 million dollars."
        onyx = companies.get("Onyx Industries")
        assert onyx is not None
        assert onyx.revenue == [4880, 4521, 204, 457], (
            f"Onyx Industries revenue expected [4880, 4521, 204, 457], got {onyx.revenue}"
        )

        # Pyra Dynamics has a single REVENUE line:
        # "Pyra Dynamics REVENUE: first quarter $4,356 million; Q2 $1,898 million; third quarter 138M; fourth quarter $3,607M."
        pyra = companies.get("Pyra Dynamics")
        assert pyra is not None
        assert pyra.revenue == [4356, 1898, 138, 3607], (
            f"Pyra Dynamics revenue expected [4356, 1898, 138, 3607], got {pyra.revenue}"
        )

        # Viro Works has a single REVENUE line
        viro = companies.get("Viro Works")
        assert viro is not None
        assert viro.revenue == [3243, 4697, 745, 3013], (
            f"Viro Works revenue expected [3243, 4697, 745, 3013], got {viro.revenue}"
        )

        # Volt Sciences has a single REVENUE line
        volt = companies.get("Volt Sciences")
        assert volt is not None
        assert volt.revenue == [864, 2503, 4584, 155], (
            f"Volt Sciences revenue expected [864, 2503, 4584, 155], got {volt.revenue}"
        )

    def test_sample2_single_revenue_companies(self, pipeline_results):
        """Sample 2: spot-check revenue for companies with single revenue lines."""
        companies = pipeline_results[2]["companies"]

        # Zeno Flow has a single revenue line
        zeno_flow = companies.get("Zeno Flow")
        assert zeno_flow is not None
        assert zeno_flow.revenue == [4853, 4349, 4551, 4547], (
            f"Zeno Flow revenue expected [4853, 4349, 4551, 4547], got {zeno_flow.revenue}"
        )

        # Rune Pulse has a single revenue line
        rune = companies.get("Rune Pulse")
        assert rune is not None
        assert rune.revenue == [3218, 4968, 1519, 2671], (
            f"Rune Pulse revenue expected [3218, 4968, 1519, 2671], got {rune.revenue}"
        )

    def test_sample3_single_revenue_companies(self, pipeline_results):
        """Sample 3: spot-check revenue for companies with single revenue lines."""
        companies = pipeline_results[3]["companies"]

        # Aero Bio actually has duplicate FINANCIALS lines (lines 166 and 210),
        # so first-write-wins keeps the first values: Q2=4844, Q3=3150
        aero = companies.get("Aero Bio")
        assert aero is not None
        assert aero.revenue == [4744, 4844, 3150, 1557], (
            f"Aero Bio revenue expected [4744, 4844, 3150, 1557], got {aero.revenue}"
        )

        # Opti Matrix
        opti = companies.get("Opti Matrix")
        assert opti is not None
        assert opti.revenue == [1491, 2560, 3105, 4323], (
            f"Opti Matrix revenue expected [1491, 2560, 3105, 4323], got {opti.revenue}"
        )


class TestStandardFormatParsing:
    """For samples 0-3 (non-transcript format), verify that ENTITY/FILING lines
    correctly populate company sector, HQ city, HQ country.

    **Validates: Requirements 3.3, 3.4**
    """

    def test_sample0_company_metadata(self, pipeline_results):
        """Sample 0: verify ENTITY/FILING parsed fields for known companies."""
        companies = pipeline_results[0]["companies"]

        # ENTITY: Zeta Cloud / Zeta | Solar Education | Haverston, Stormveil
        zc = companies.get("Zeta Cloud")
        assert zc is not None
        assert zc.sector == "Solar Education"
        assert zc.hq_city == "Haverston"
        assert zc.hq_country == "Stormveil"

        # FILING: Coda Data (DBA: CD) | Sector: Hydro Engineering | Jurisdiction: Jinsworth, Grenvald
        cd = companies.get("Coda Data")
        assert cd is not None
        assert cd.sector == "Hydro Engineering"
        assert cd.hq_city == "Jinsworth"
        assert cd.hq_country == "Grenvald"
        assert cd.dba == "CD"

        # ENTITY: Onyx Industries / Onyx | Electro Engineering | Rosemount, Caltheon
        onyx = companies.get("Onyx Industries")
        assert onyx is not None
        assert onyx.sector == "Electro Engineering"
        assert onyx.hq_city == "Rosemount"
        assert onyx.hq_country == "Caltheon"

    def test_sample0_executive_data(self, pipeline_results):
        """Sample 0: verify CEO names parsed from EXECUTIVE/OFFICER lines."""
        companies = pipeline_results[0]["companies"]

        # Xeno Logic EXECUTIVE: Ward Tran, Co-CEO
        xeno = companies.get("Xeno Logic")
        assert xeno is not None
        assert xeno.ceo_last_name == "Tran"

        # Prim Industries EXECUTIVE: Nolan Holt, Co-CEO
        prim = companies.get("Prim Industries")
        assert prim is not None
        assert prim.ceo_last_name == "Holt"

        # Byte Wave: BW OFFICER: CEO Jules Archer
        bw = companies.get("Byte Wave")
        assert bw is not None
        assert bw.ceo_last_name == "Archer"

    def test_sample0_ratios_and_employees(self, pipeline_results):
        """Sample 0: verify D/E ratio, satisfaction, and employee counts."""
        companies = pipeline_results[0]["companies"]

        # Pyra Dynamics RATIOS: D/E 3.3 | Satisfaction 8.2/10 | Employees approximately 41819.
        pyra = companies.get("Pyra Dynamics")
        assert pyra is not None
        assert pyra.de_ratio == pytest.approx(3.3, abs=0.01)
        assert pyra.satisfaction == pytest.approx(8.2, abs=0.01)
        assert pyra.employees == 41819

        # Coda Data RATIOS: D/E 1.0 | Satisfaction 9.7/10 | Employees 67,781.
        cd = companies.get("Coda Data")
        assert cd is not None
        assert cd.de_ratio == pytest.approx(1.0, abs=0.01)
        assert cd.satisfaction == pytest.approx(9.7, abs=0.01)
        assert cd.employees == 67781

    def test_sample2_company_metadata(self, pipeline_results):
        """Sample 2: verify ENTITY/FILING parsed fields."""
        companies = pipeline_results[2]["companies"]

        # Zyra Net: city=Thornfield, country=Grenvald
        zyra_net = companies.get("Zyra Net")
        assert zyra_net is not None
        assert zyra_net.hq_city == "Thornfield"
        assert zyra_net.hq_country == "Grenvald"

        # Torq Matrix: city=Glenspire, country=Istara
        torq_matrix = companies.get("Torq Matrix")
        assert torq_matrix is not None
        assert torq_matrix.hq_city == "Glenspire"
        assert torq_matrix.hq_country == "Istara"

    def test_sample3_company_metadata(self, pipeline_results):
        """Sample 3: verify ENTITY/FILING parsed fields."""
        companies = pipeline_results[3]["companies"]

        # Aero Bio: sector=Sonic Aerospace, city=Elsmere, country=Westmark
        aero = companies.get("Aero Bio")
        assert aero is not None
        assert aero.hq_city == "Elsmere"
        assert aero.hq_country == "Westmark"

    @pytest.mark.parametrize("sample_idx", [0, 2, 3])
    def test_standard_format_parses_companies(self, pipeline_results, sample_idx):
        """Standard format samples should parse a non-zero number of companies."""
        companies = pipeline_results[sample_idx]["companies"]
        assert len(companies) > 0, (
            f"Sample {sample_idx}: expected parsed companies, got 0"
        )
        # At least some companies should have HQ data
        with_city = sum(1 for c in companies.values() if c.hq_city)
        assert with_city > 0, (
            f"Sample {sample_idx}: no companies have HQ city data"
        )


class TestQuestionAnsweringPreservation:
    """For questions whose answers are NOT affected by the bugs, verify correctness.

    The bugs affect: revenue data (overwrite), acrostic refs, and transcript parsing.
    Questions that depend on non-revenue, non-transcript data should be stable.

    **Validates: Requirements 3.4, 3.6**
    """

    def test_sample0_non_revenue_answers(self, pipeline_results):
        """Sample 0: verify answers to questions not affected by revenue overwrite."""
        answers = pipeline_results[0]["answers"]

        # Q1 (idx 0): "revenue volatility" — depends on revenue but Onyx Industries
        # has a single REVENUE line, so its data is correct
        assert answers[0] == "Onyx Industries"

        # Q5 (idx 4): "public companies, lowest D/E" — depends on D/E ratio, not revenue
        assert answers[4] == "Prim Industries"

        # Q6 (idx 5): "most recent IPO" — depends on IPO year, not revenue
        assert answers[5] == "Juno Capital"

        # Q8 (idx 7): "Agri sector, founded earliest" — depends on sector/founded
        assert answers[7] == "Juno Dynamics"

        # Q10 (idx 9): "decline in Q3, highest D/E" — Mira Energy has D/E=4.7 vs Rune Tech's 3.8
        assert answers[9] == "Mira Energy"

    def test_sample2_non_revenue_answers(self, pipeline_results):
        """Sample 2: verify answers to questions not affected by bugs."""
        answers = pipeline_results[2]["answers"]

        # Q1 (idx 0): "public companies, positive growth every quarter, fewest employees"
        assert answers[0] == "Zeno Flow"

        # Q2 (idx 1): "founded in 1990s with positive growth every quarter"
        assert answers[1] == "Zeno Flow"

        # Q4 (idx 3): "privately held, best satisfaction" — Sero Solutions is public; Opti Tech (satisfaction=5.9) is correct
        assert answers[3] == "Opti Tech"

        # Q9 (idx 8): "oldest company in Bio sector"
        assert answers[8] == "Dyna Prime"

    def test_sample3_non_revenue_answers(self, pipeline_results):
        """Sample 3: verify answers to questions not affected by bugs."""
        answers = pipeline_results[3]["answers"]

        # Q2 (idx 1): "headquartered in Haverston, highest Q2 revenue"
        assert answers[1] == "Flux Works"

        # Q7 (idx 6): "privately held, highest satisfaction"
        assert answers[6] == "Mira Data"

        # Q9 (idx 8): "second-earliest founding year" — Cryo Wave (1974) is second after Lux Works (1970)
        assert answers[8] == "Cryo Wave"
