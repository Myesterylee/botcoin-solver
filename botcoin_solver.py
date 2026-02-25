#!/usr/bin/env python3
"""BOTCOIN Mining Challenge Solver v2 — Deterministic Engine

Zero LLM dependency. All parsing, filtering, sorting, comparison done in pure Python.
Multi-pass document parsing with aggressive normalization.
Question classification via regex → deterministic computation.

Usage:
    echo '{"doc":"...","questions":[...],"constraints":[...]}' | python botcoin_solver.py
    python botcoin_solver.py --test   # run self-tests
"""

import json
import sys
import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CompanyData:
    """All known data about a single fictional company."""
    name: str = ""              # Full company name (e.g., "Torq Hub")
    dba: str = ""               # DBA abbreviation (e.g., "TH")
    short_name: str = ""        # First word / slash-alias (e.g., "Torq")
    hq_city: str = ""
    hq_country: str = ""
    sector: str = ""
    ceo_full_name: str = ""
    ceo_last_name: str = ""
    founded_year: int = 0
    is_public: bool = False
    ipo_year: int = 0           # 0 if private
    employees: int = 0
    de_ratio: float = 0.0
    satisfaction: float = 0.0
    revenue: List[int] = field(default_factory=lambda: [0, 0, 0, 0])   # [Q1,Q2,Q3,Q4] millions
    growth: List[Optional[float]] = field(default_factory=lambda: [None, None, None, None])  # percentages

    def total_revenue(self) -> int:
        """Sum of all four quarters."""
        return sum(self.revenue)

    def has_revenue(self) -> bool:
        """True if any quarter has non-zero revenue."""
        return any(r > 0 for r in self.revenue)

    def revenue_volatility(self) -> int:
        """Max quarterly revenue minus min quarterly revenue."""
        if not self.has_revenue():
            return 0
        vals = [r for r in self.revenue if r > 0]
        if not vals:
            return 0
        return max(self.revenue) - min(self.revenue)

    def all_growth_positive(self) -> bool:
        """True if all four quarters have strictly positive growth."""
        return all(g is not None and g > 0 for g in self.growth)

    def avg_growth(self) -> float:
        """Average of the four quarterly growth rates."""
        valid = [g for g in self.growth if g is not None]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)


def dbg(msg: str):
    """Print debug info to stderr."""
    print(msg, file=sys.stderr)


# =============================================================================
# REVENUE / FINANCIAL PARSING UTILITIES
# =============================================================================

def parse_revenue_amount(text: str) -> int:
    """Parse a revenue amount string to integer millions.

    Handles all known formats:
        $4,356 million  → 4356
        $4,356M         → 4356
        4,356M          → 4356
        $4.88 billion   → 4880
        $4.88B          → 4880
        2475 million dollars → 2475
        $1,234          → 1234  (bare number assumed millions)

    Returns 0 if unparseable.
    """
    if not text:
        return 0
    t = text.strip()

    # Remove leading modifiers
    t = re.sub(
        r'^(close\s+to|approximately|just\s+under|just\s+over|nearly|roughly|about|around|over|under)\s+',
        '', t, flags=re.IGNORECASE
    ).strip()

    # Try billion patterns first (higher priority)
    # "$4.88 billion" or "4.88B" or "$4.88B" or "4.88 billion"
    m = re.search(r'\$?([\d,]+(?:\.\d+)?)\s*(?:billion|B)\b', t, re.IGNORECASE)
    if m:
        val = float(m.group(1).replace(',', ''))
        return int(round(val * 1000))

    # "$4,356 million" or "$4,356M" or "4356 million dollars" or "4356M"
    m = re.search(r'\$?([\d,]+(?:\.\d+)?)\s*(?:million(?:\s+dollars)?|M)\b', t, re.IGNORECASE)
    if m:
        val = float(m.group(1).replace(',', ''))
        return int(round(val))

    # "2475 million dollars" (without $ sign, number then "million dollars")
    m = re.search(r'([\d,]+(?:\.\d+)?)\s+million\s+dollars', t, re.IGNORECASE)
    if m:
        val = float(m.group(1).replace(',', ''))
        return int(round(val))

    # Bare "$4,356" — assume millions
    m = re.search(r'\$?([\d,]+(?:\.\d+)?)', t)
    if m:
        val = float(m.group(1).replace(',', ''))
        if val > 0:
            return int(round(val))

    return 0


def parse_growth_rate(text: str) -> Optional[float]:
    """Parse a growth rate string to a float percentage.

    Handles: +25%, -12%, 0%, (+25%), (-12%)
    Returns None if unparseable.
    """
    if not text:
        return None
    m = re.search(r'([+-]?\d+(?:\.\d+)?)\s*%', text)
    if m:
        return float(m.group(1))
    return None


def parse_employee_count(text: str) -> int:
    """Parse employee count from text.

    Handles modifiers: approximately, nearly, just over, just under,
    close to, roughly, about, around.
    Handles comma-separated numbers: 41,819
    """
    if not text:
        return 0
    t = text.strip()
    t = re.sub(
        r'(approximately|nearly|just\s+over|just\s+under|close\s+to|roughly|about|around)\s+',
        '', t, flags=re.IGNORECASE
    ).strip()
    m = re.search(r'([\d,]+)', t)
    if m:
        return int(m.group(1).replace(',', ''))
    return 0


def next_prime(n: int) -> int:
    """Return the smallest prime >= n."""
    if n <= 2:
        return 2
    candidate = n
    while True:
        if is_prime(candidate):
            return candidate
        candidate += 1


def is_prime(n: int) -> bool:
    """Primality test."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def get_initials(name: str) -> str:
    """Get uppercase initials of a company name.

    "Xeno Logic" → "XL"
    "Torq Hub" → "TH"
    """
    words = name.strip().split()
    return ''.join(w[0].upper() for w in words if w)


# =============================================================================
# QUARTER LABEL MAPPING
# =============================================================================

QUARTER_MAP = {
    'q1': 0, 'q2': 1, 'q3': 2, 'q4': 3,
    'first quarter': 0, 'second quarter': 1, 'third quarter': 2, 'fourth quarter': 3,
    '1st quarter': 0, '2nd quarter': 1, '3rd quarter': 2, '4th quarter': 3,
    'opening quarter': 0, 'closing quarter': 3,
    'mid-spring quarter': 1, 'follow-on quarter': 1,
    'pre-close quarter': 2, 'late-year quarter': 2,
    'year-end quarter': 3,
    "year's first quarter": 0,
    "the year's first quarter": 0,
    "the mid-spring quarter": 1,
    "the follow-on quarter": 1,
    "the pre-close quarter": 2,
    "the late-year quarter": 2,
    "the year-end quarter": 3,
    "the closing quarter": 3,
    "the opening quarter": 0,
}


def identify_quarter(text: str) -> Optional[int]:
    """Identify which quarter (0-3) a text segment refers to.

    Returns None if no quarter label found.
    """
    t = text.strip().lower()

    # Try longest matches first to avoid partial matches
    for label in sorted(QUARTER_MAP.keys(), key=len, reverse=True):
        if label in t:
            return QUARTER_MAP[label]

    return None


# =============================================================================
# COUNTERFACTUAL DETECTION
# =============================================================================

COUNTERFACTUAL_PATTERNS = [
    re.compile(r'\bif\b.*\bhad\s+pursued\b', re.IGNORECASE),
    re.compile(r'\bcounterfactual\b', re.IGNORECASE),
    re.compile(r'\bnever\s+materialized\b', re.IGNORECASE),
    re.compile(r'\brumored\s+merger\b', re.IGNORECASE),
    re.compile(r'\bchatter\s+suggested\b', re.IGNORECASE),
    re.compile(r'\bwalked\s+back\b', re.IGNORECASE),
    re.compile(r'\brumored\s+acquisition\b', re.IGNORECASE),
    re.compile(r'\bhypothetical\b', re.IGNORECASE),
    re.compile(r'\bspeculative\b', re.IGNORECASE),
    re.compile(r'\bnever\s+came\s+to\s+pass\b', re.IGNORECASE),
    re.compile(r'\bnever\s+happened\b', re.IGNORECASE),
    re.compile(r'\bpurely\s+theoretical\b', re.IGNORECASE),
    re.compile(r'\bcould\s+have\s+looked\s+like\b', re.IGNORECASE),
    re.compile(r'\bsupposed\s+acquisition\b', re.IGNORECASE),
    re.compile(r'\bclaim\s+was\s+later\b', re.IGNORECASE),
    re.compile(r'\brunaway\s+scenario\b', re.IGNORECASE),
    re.compile(r'\bwhat[\s-]+if\b', re.IGNORECASE),
    re.compile(r'\bfantasy\s+figure\b', re.IGNORECASE),
    re.compile(r'\bnever\s+took\s+place\b', re.IGNORECASE),
    re.compile(r'\bnever\s+finalized\b', re.IGNORECASE),
    re.compile(r'\bnever\s+executed\b', re.IGNORECASE),
    re.compile(r'\bdebunked\b', re.IGNORECASE),
    re.compile(r'\bdisproven\b', re.IGNORECASE),
    re.compile(r'\bwas\s+later\s+retracted\b', re.IGNORECASE),
    re.compile(r'\bwas\s+later\s+denied\b', re.IGNORECASE),
    re.compile(r'\bwas\s+never\s+confirmed\b', re.IGNORECASE),
    re.compile(r'\bimaginary\b', re.IGNORECASE),
    re.compile(r'\bfictitious\s+scenario\b', re.IGNORECASE),
]


def is_counterfactual(line: str) -> bool:
    """Check if a line contains counterfactual/hypothetical data that should be ignored."""
    for pat in COUNTERFACTUAL_PATTERNS:
        if pat.search(line):
            return True
    return False



# =============================================================================
# DOCUMENT PARSER
# =============================================================================

class DocumentParser:
    """Multi-pass parser for BOTCOIN challenge documents.

    Handles all known line formats: ENTITY, FILING, EXECUTIVE/OFFICER/management,
    RATIOS/DISCLOSURE, REVENUE/FINANCIALS, and transcript (PANELIST/ANALYST/MODERATOR).

    Uses aggressive name resolution to link DBA abbreviations, short names,
    and full names to the same company record.
    """

    def __init__(self, doc: str, company_names: List[str]):
        """
        Args:
            doc: The raw document text.
            company_names: List of known company names from the challenge metadata
                           (often extracted from questions or provided separately).
        """
        self.doc = doc
        self.lines = doc.split('\n')
        self.known_names = company_names
        self.companies: Dict[str, CompanyData] = {}
        # Lookup: lowercase alias → canonical company name
        self.name_lookup: Dict[str, str] = {}

    def parse(self) -> Dict[str, CompanyData]:
        """Run all parsing passes and return company data dict keyed by canonical name."""
        # Pass 1: Extract company identities (ENTITY/FILING lines)
        self._pass_identity()
        # Pass 2: Build comprehensive name lookup
        self._build_name_lookup()
        # Pass 3: Parse all data lines
        self._pass_data()
        # Pass 4: Parse transcript-format lines
        self._pass_transcript()

        dbg(f"\n=== PARSED {len(self.companies)} COMPANIES ===")
        for name, c in sorted(self.companies.items()):
            dbg(f"  {name}: sector={c.sector}, city={c.hq_city}, country={c.hq_country}, "
                f"CEO={c.ceo_full_name}, founded={c.founded_year}, public={c.is_public}, "
                f"ipo={c.ipo_year}, emp={c.employees}, D/E={c.de_ratio}, sat={c.satisfaction}, "
                f"rev={c.revenue}, growth={c.growth}")

        return self.companies

    def _ensure_company(self, name: str) -> CompanyData:
        """Get or create a CompanyData record for the given canonical name."""
        if name not in self.companies:
            self.companies[name] = CompanyData(name=name)
        return self.companies[name]

    def _register_alias(self, alias: str, canonical: str):
        """Register a name alias in the lookup table."""
        if alias and canonical:
            self.name_lookup[alias.strip().lower()] = canonical

    def _resolve_company(self, text: str) -> Optional[str]:
        """Resolve a text prefix to a canonical company name.

        Priority: exact full name > DBA code > short name > fuzzy match.
        Also handles possessive forms (e.g., "Giga Cloud's" → "Giga Cloud").
        """
        t = text.strip()
        # Strip possessive suffix
        if t.endswith("'s"):
            t = t[:-2]
        tl = t.lower()

        # Exact match
        if tl in self.name_lookup:
            return self.name_lookup[tl]

        # Try progressively shorter prefixes
        words = t.split()
        for i in range(len(words), 0, -1):
            prefix = ' '.join(words[:i]).lower()
            if prefix in self.name_lookup:
                return self.name_lookup[prefix]

        # Try just the first word (short name)
        if words:
            first = words[0].lower()
            if first in self.name_lookup:
                return self.name_lookup[first]

        return None

    # ----- Pass 1: Identity lines -----

    def _pass_identity(self):
        """Extract company names, sectors, HQ from ENTITY, FILING, and non-standard identity lines."""
        for line in self.lines:
            line = line.strip()
            if not line:
                continue
            if is_counterfactual(line):
                continue

            # ENTITY: Full Name / Short | Sector | City, Country
            m = re.match(
                r'ENTITY:\s*(.+?)\s*/\s*(\w+)\s*\|\s*(.+?)\s*\|\s*(.+?),\s*(.+?)$',
                line
            )
            if m:
                full_name = m.group(1).strip()
                short = m.group(2).strip()
                sector = m.group(3).strip()
                city = m.group(4).strip()
                country = m.group(5).strip()
                c = self._ensure_company(full_name)
                c.short_name = short
                c.sector = sector
                c.hq_city = city
                c.hq_country = country
                continue

            # FILING: Full Name (DBA: XX) | Sector: ... | Jurisdiction: City, Country
            m = re.match(
                r'FILING:\s*(.+?)\s*\(DBA:\s*(\w+)\)\s*\|\s*Sector:\s*(.+?)\s*\|\s*Jurisdiction:\s*(.+?),\s*(.+?)$',
                line
            )
            if m:
                full_name = m.group(1).strip()
                dba = m.group(2).strip()
                sector = m.group(3).strip()
                city = m.group(4).strip()
                country = m.group(5).strip()
                c = self._ensure_company(full_name)
                c.dba = dba
                c.sector = sector
                c.hq_city = city
                c.hq_country = country
                continue

            # "COMPANY is a SECTOR company based in CITY, COUNTRY. People ... call it DBA ..."
            m = re.match(
                r'^(.+?)\s+is\s+a\s+(.+?)\s+company\s+based\s+in\s+(\w+),\s*(\w+)\.\s*People\s+.*call\s+it\s+(\w+)',
                line, re.IGNORECASE
            )
            if m:
                full_name = m.group(1).strip()
                sector = m.group(2).strip()
                city = m.group(3).strip()
                country = m.group(4).strip()
                dba = m.group(5).strip()
                c = self._ensure_company(full_name)
                c.sector = sector
                c.hq_city = city
                c.hq_country = country
                c.dba = dba
                self._register_alias(dba, full_name)
                continue

            # "From CITY (COUNTRY), COMPANY operates in SECTOR. In internal notes ... as SHORT."
            m = re.match(
                r'^From\s+(\w+)\s*\((\w+)\),\s*(.+?)\s+operates\s+in\s+(.+?)\.\s*In\s+internal\s+notes\s+.*(?:as|called)\s+(\w+)',
                line, re.IGNORECASE
            )
            if m:
                city = m.group(1).strip()
                country = m.group(2).strip()
                full_name = m.group(3).strip()
                sector = m.group(4).strip()
                short_name = m.group(5).strip()
                c = self._ensure_company(full_name)
                c.hq_city = city
                c.hq_country = country
                c.sector = sector
                c.short_name = short_name
                self._register_alias(short_name, full_name)
                continue

    def _build_name_lookup(self):
        """Build comprehensive alias → canonical name lookup table."""
        for name, c in self.companies.items():
            self._register_alias(name, name)
            if c.dba:
                self._register_alias(c.dba, name)
            if c.short_name:
                self._register_alias(c.short_name, name)
            # Also register first word as short name if not already set
            words = name.split()
            if len(words) > 1:
                if not c.short_name:
                    c.short_name = words[0]
                self._register_alias(words[0], name)

        # Auto-generate DBA codes from company name initials (e.g., "Halo Sciences" → "HS")
        # Only register if the initials code is not already taken by another company
        for name, c in self.companies.items():
            words = name.split()
            if len(words) >= 2:
                initials = ''.join(w[0].upper() for w in words if w)
                initials_lower = initials.lower()
                if initials_lower not in self.name_lookup:
                    self._register_alias(initials, name)

        # Also register known names from challenge metadata
        for kn in self.known_names:
            kn_stripped = kn.strip()
            knl = kn_stripped.lower()
            if knl not in self.name_lookup:
                # Try to match to existing company
                for cname in self.companies:
                    if knl == cname.lower() or kn_stripped in cname:
                        self._register_alias(kn_stripped, cname)
                        break
            # Also ensure known names create companies if not already present
            if knl not in self.name_lookup and kn_stripped:
                self._ensure_company(kn_stripped)
                self._register_alias(kn_stripped, kn_stripped)

    def _resolve_line_prefix(self, line: str) -> Tuple[Optional[str], str]:
        """Given a line like 'TH FINANCIALS: ...', resolve the prefix to a company name.

        Returns (canonical_name, remainder_of_line) or (None, line).
        """
        # Try patterns: "PREFIX KEYWORD:" where KEYWORD is a known section type
        keywords = [
            'REVENUE', 'FINANCIALS', 'RATIOS', 'DISCLOSURE', 'EXECUTIVE',
            'OFFICER', 'MANAGEMENT', 'management', 'officer', 'executive',
            'GROWTH', 'DATA', 'FISCAL', 'QUARTERLY', 'ANNUAL',
            'ratios', 'disclosure', 'revenue', 'financials',
        ]

        for kw in keywords:
            # Pattern: "PREFIX KEYWORD:" (prefix can be multi-word)
            pat = re.compile(r'^(.+?)\s+' + re.escape(kw) + r'\s*:\s*(.*)$', re.IGNORECASE)
            m = pat.match(line)
            if m:
                prefix = m.group(1).strip()
                remainder = m.group(2).strip()
                resolved = self._resolve_company(prefix)
                if resolved:
                    return resolved, remainder

        return None, line

    # ----- Pass 2: Data lines -----

    def _pass_data(self):
        """Parse executive, ratios, revenue, and other data lines."""
        for line in self.lines:
            line = line.strip()
            if not line:
                continue
            if is_counterfactual(line):
                continue

            # Try to parse as executive/officer line
            self._try_parse_executive(line)
            # Try to parse as ratios/disclosure line
            self._try_parse_ratios(line)
            # Try to parse as revenue/financials line
            self._try_parse_revenue(line)
            # Try to parse non-standard format lines
            self._try_parse_tape_line(line)
            self._try_parse_capital_structure(line)
            self._try_parse_leverage_line(line)
            self._try_parse_at_company_line(line)
            self._try_parse_for_company_revenue(line)

    def _try_parse_executive(self, line: str):
        """Parse executive/officer/management lines."""
        # Pattern: "COMPANY EXECUTIVE: Title Name | Founded YYYY | went public in YYYY"
        # Also: "COMPANY OFFICER: ..." and "COMPANY management: ..."
        m = re.match(
            r'^(.+?)\s+(?:EXECUTIVE|OFFICER|management)\s*:\s*(.+)$',
            line, re.IGNORECASE
        )
        if not m:
            return

        prefix = m.group(1).strip()
        remainder = m.group(2).strip()
        company_name = self._resolve_company(prefix)
        if not company_name:
            return

        c = self._ensure_company(company_name)

        # Parse CEO name and title
        # Formats:
        #   "CEO Kael Pryce" / "Co-CEO Kael Pryce" / "President & CEO Kael Pryce"
        #   "Managing Director Ulric Mercer"
        #   "Founder & CEO Kael Pryce"
        #   "Kael Pryce, CEO" / "Kael Pryce, Co-CEO"
        ceo_part = remainder.split('|')[0].strip() if '|' in remainder else remainder.split('.')[0].strip()
        # Remove "Founded..." and everything after from ceo_part
        ceo_part = re.split(r'\bFounded\b', ceo_part, flags=re.IGNORECASE)[0].strip()
        ceo_part = ceo_part.rstrip('|').rstrip(';').strip()

        ceo_name = self._extract_ceo_name(ceo_part)
        if ceo_name:
            c.ceo_full_name = ceo_name
            c.ceo_last_name = ceo_name.split()[-1] if ceo_name.split() else ""

        # Parse founded year
        fm = re.search(r'[Ff]ounded\s+(?:in\s+)?(\d{4})', remainder)
        if fm:
            c.founded_year = int(fm.group(1))

        # Parse IPO status
        if re.search(r'went\s+public\s+in\s+(\d{4})', remainder, re.IGNORECASE):
            ipo_m = re.search(r'went\s+public\s+in\s+(\d{4})', remainder, re.IGNORECASE)
            c.is_public = True
            c.ipo_year = int(ipo_m.group(1))
        elif re.search(r'remains?\s+(?:privately\s+held|private)', remainder, re.IGNORECASE):
            c.is_public = False
            c.ipo_year = 0

        # Parse employee count from this line
        emp_m = re.search(
            r'(?:employs?\s+(?:nearly|approximately|roughly|about|close\s+to|just\s+over|just\s+under|around)?\s*)([\d,]+)\s*(?:people|employees|staff|workers)',
            remainder, re.IGNORECASE
        )
        if emp_m:
            c.employees = int(emp_m.group(1).replace(',', ''))
        else:
            emp_m2 = re.search(r'([\d,]+)\s*(?:people|employees|staff|workers)', remainder, re.IGNORECASE)
            if emp_m2:
                c.employees = int(emp_m2.group(1).replace(',', ''))

    def _extract_ceo_name(self, text: str) -> str:
        """Extract CEO name from various title formats.

        Returns just "FirstName LastName".
        """
        t = text.strip().rstrip(',').rstrip(';').strip()

        # "FirstName LastName, Title" format
        m = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*,\s*(.+)$', t)
        if m:
            name_part = m.group(1).strip()
            title_part = m.group(2).strip().lower()
            if any(kw in title_part for kw in ['ceo', 'chief executive', 'managing director', 'president', 'founder']):
                return name_part

        # "Title FirstName LastName" format
        # Remove title prefixes
        title_patterns = [
            r'^(?:President\s*&\s*)?(?:Co-)?CEO\s+',
            r'^(?:Founder\s*&\s*)?(?:Co-)?CEO\s+',
            r'^Chief\s+Executive(?:\s+Officer)?\s+',
            r'^Managing\s+Director\s+',
            r'^President\s+(?:&\s+CEO\s+)?',
            r'^Founder\s+(?:&\s+CEO\s+)?',
            r'^Director\s+',
            r'^(?:Co-)?Chief\s+Executive\s+',
        ]
        for pat in title_patterns:
            m = re.match(pat, t, re.IGNORECASE)
            if m:
                name = t[m.end():].strip()
                # Clean trailing titles
                name = re.sub(r'\s*[,;|].*$', '', name)
                if name and re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+', name):
                    return name.split(',')[0].strip()

        # Fallback: look for "FirstName LastName" pattern (two capitalized words)
        m = re.search(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b', t)
        if m:
            return m.group(1)

        return t

    def _try_parse_ratios(self, line: str):
        """Parse ratios/disclosure lines for D/E, satisfaction, employees."""
        # Check if line contains ratio keywords
        if not re.search(r'(?:RATIOS|DISCLOSURE|D/E|[Dd]ebt.to.equity|[Ss]atisfaction)', line):
            return

        company_name, remainder = self._resolve_line_prefix(line)
        if not company_name:
            # Try to find company name at start of line before D/E or satisfaction
            m = re.match(r'^(.+?)\s+(?:D/E|[Dd]ebt)', line)
            if m:
                company_name = self._resolve_company(m.group(1).strip())
                remainder = line[m.start(1):]
            if not company_name:
                return

        c = self._ensure_company(company_name)

        # D/E ratio
        de_m = re.search(r'(?:D/E|[Dd]ebt[\s-]*to[\s-]*equity)\s*:?\s*(\d+(?:\.\d+)?)', remainder)
        if de_m:
            c.de_ratio = float(de_m.group(1))

        # Satisfaction
        sat_m = re.search(r'[Ss]atisfaction\s*:?\s*(\d+(?:\.\d+)?)(?:\s*/\s*10)?', remainder)
        if sat_m:
            c.satisfaction = float(sat_m.group(1))

        # Employees (in ratios line)
        emp_m = re.search(
            r'[Ee]mployees?\s+(?:approximately|nearly|roughly|about|close\s+to|just\s+over|just\s+under|around)?\s*([\d,]+)',
            remainder
        )
        if emp_m:
            c.employees = int(emp_m.group(1).replace(',', ''))


    def _try_parse_revenue(self, line: str):
        """Parse revenue/financials lines for quarterly revenue and growth."""
        # Check if line contains revenue keywords or quarter references with dollar amounts
        has_revenue_kw = re.search(
            r'(?:REVENUE|FINANCIALS|FISCAL|QUARTERLY)',
            line, re.IGNORECASE
        )
        has_quarter_data = re.search(
            r'(?:Q[1-4]|first\s+quarter|second\s+quarter|third\s+quarter|fourth\s+quarter|opening\s+quarter|closing\s+quarter)',
            line, re.IGNORECASE
        )
        if not (has_revenue_kw or has_quarter_data):
            return

        # Must also have some dollar/million/billion indicator
        has_money = re.search(r'(?:\$|million|billion|M\b|B\b)', line, re.IGNORECASE)
        if not has_money:
            return

        company_name, remainder = self._resolve_line_prefix(line)
        if not company_name:
            return

        c = self._ensure_company(company_name)

        # Parse quarterly revenue and growth from the remainder
        self._extract_quarterly_data(c, remainder)

        # Also check for GROWTH section in the same line
        growth_m = re.search(r'GROWTH\s*:\s*(.+)$', line, re.IGNORECASE)
        if growth_m:
            self._extract_growth_data(c, growth_m.group(1))

    def _extract_quarterly_data(self, c: CompanyData, text: str):
        """Extract quarterly revenue and inline growth from a text segment.

        Handles complex formats like:
        "first quarter $4,356 million; Q2 $1,898 million; third quarter 138M; fourth quarter $3,607M"
        "opening quarter close to 2475 million dollars (+36%), Q2 1,990 million dollars (-28%)"
        """
        # Strategy: split by known quarter labels, then parse each segment
        # First, find all quarter markers and their positions
        quarter_markers = []

        # Build a comprehensive regex for quarter labels
        quarter_patterns = [
            (r"\bQ1\b", 0), (r"\bQ2\b", 1), (r"\bQ3\b", 2), (r"\bQ4\b", 3),
            (r"\bfirst\s+quarter\b", 0), (r"\bsecond\s+quarter\b", 1),
            (r"\bthird\s+quarter\b", 2), (r"\bfourth\s+quarter\b", 3),
            (r"\bopening\s+quarter\b", 0), (r"\bclosing\s+quarter\b", 3),
            (r"\bmid-spring\s+quarter\b", 1), (r"\bfollow-on\s+quarter\b", 1),
            (r"\bpre-close\s+quarter\b", 2), (r"\blate-year\s+quarter\b", 2),
            (r"\byear-end\s+quarter\b", 3), (r"\byear's\s+first\s+quarter\b", 0),
            (r"\b1st\s+quarter\b", 0), (r"\b2nd\s+quarter\b", 1),
            (r"\b3rd\s+quarter\b", 2), (r"\b4th\s+quarter\b", 3),
        ]

        for pat, qi in quarter_patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                quarter_markers.append((m.start(), m.end(), qi))

        # Sort by position
        quarter_markers.sort(key=lambda x: x[0])

        # For each quarter marker, extract the text segment until the next marker
        for idx, (start, end, qi) in enumerate(quarter_markers):
            # Segment goes from end of this marker to start of next marker (or end of text)
            if idx + 1 < len(quarter_markers):
                segment = text[end:quarter_markers[idx + 1][0]]
            else:
                segment = text[end:]

            # Extract revenue amount from segment
            rev = self._parse_revenue_from_segment(segment)
            if rev > 0:
                if c.revenue[qi] == 0:
                    c.revenue[qi] = rev

            # Extract inline growth rate
            growth = self._parse_inline_growth(segment)
            if growth is not None:
                if c.growth[qi] is None:
                    c.growth[qi] = growth

    def _parse_revenue_from_segment(self, segment: str) -> int:
        """Parse revenue amount from a quarter segment.

        Handles all known formats including modifiers.
        """
        s = segment.strip()
        # Remove leading separators
        s = re.sub(r'^[\s,;:/\-]+', '', s).strip()

        # Try billion first
        m = re.search(r'\$?([\d,]+(?:\.\d+)?)\s*(?:billion|B)\b', s, re.IGNORECASE)
        if m:
            val = float(m.group(1).replace(',', ''))
            return int(round(val * 1000))

        # Try million/M patterns
        m = re.search(r'\$?([\d,]+(?:\.\d+)?)\s*(?:million(?:\s+dollars)?|M)\b', s, re.IGNORECASE)
        if m:
            val = float(m.group(1).replace(',', ''))
            return int(round(val))

        # Try "NUMBER million dollars" with modifiers
        m = re.search(
            r'(?:close\s+to|approximately|just\s+under|just\s+over|nearly|roughly|about|around)?\s*([\d,]+(?:\.\d+)?)\s+million\s+dollars',
            s, re.IGNORECASE
        )
        if m:
            val = float(m.group(1).replace(',', ''))
            return int(round(val))

        # Try bare "$NUMBER" (assumed millions)
        m = re.search(r'\$([\d,]+(?:\.\d+)?)\b', s)
        if m:
            val = float(m.group(1).replace(',', ''))
            if val > 0:
                return int(round(val))

        # Try bare "NUMBER million dollars" without modifier prefix
        m = re.search(r'([\d,]+(?:\.\d+)?)\s+million\s+dollars', s, re.IGNORECASE)
        if m:
            val = float(m.group(1).replace(',', ''))
            return int(round(val))

        return 0

    def _parse_inline_growth(self, segment: str) -> Optional[float]:
        """Extract growth rate from a revenue segment.

        Handles: (+25%), (-12%), (0%), / +25%, / -12%
        """
        # Parenthesized: (+25%) or (-12%) or (0%)
        m = re.search(r'\(\s*([+-]?\d+(?:\.\d+)?)\s*%\s*\)', segment)
        if m:
            return float(m.group(1))

        # Slash-separated: / +25% or / -12%
        m = re.search(r'/\s*([+-]?\d+(?:\.\d+)?)\s*%', segment)
        if m:
            return float(m.group(1))

        return None

    def _extract_growth_data(self, c: CompanyData, text: str):
        """Extract growth rates from a dedicated GROWTH section."""
        # Same approach as quarterly data but only looking for growth rates
        quarter_patterns = [
            (r"\bQ1\b", 0), (r"\bQ2\b", 1), (r"\bQ3\b", 2), (r"\bQ4\b", 3),
            (r"\bfirst\s+quarter\b", 0), (r"\bsecond\s+quarter\b", 1),
            (r"\bthird\s+quarter\b", 2), (r"\bfourth\s+quarter\b", 3),
            (r"\bopening\s+quarter\b", 0), (r"\bclosing\s+quarter\b", 3),
            (r"\bmid-spring\s+quarter\b", 1), (r"\bfollow-on\s+quarter\b", 1),
            (r"\bpre-close\s+quarter\b", 2), (r"\blate-year\s+quarter\b", 2),
            (r"\byear-end\s+quarter\b", 3), (r"\byear's\s+first\s+quarter\b", 0),
        ]

        quarter_markers = []
        for pat, qi in quarter_patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                quarter_markers.append((m.start(), m.end(), qi))

        quarter_markers.sort(key=lambda x: x[0])

        for idx, (start, end, qi) in enumerate(quarter_markers):
            if idx + 1 < len(quarter_markers):
                segment = text[end:quarter_markers[idx + 1][0]]
            else:
                segment = text[end:]

            m = re.search(r'([+-]?\d+(?:\.\d+)?)\s*%', segment)
            if m:
                if c.growth[qi] is None:
                    c.growth[qi] = float(m.group(1))

    # ----- Non-standard format line parsers -----

    def _try_parse_tape_line(self, line: str):
        """Parse 'DBA tape: Q1 $X / +Y%; ...' revenue lines."""
        m = re.match(r'^(\w+)\s+tape\s*:\s*(.+)$', line, re.IGNORECASE)
        if not m:
            return
        prefix = m.group(1).strip()
        remainder = m.group(2).strip()
        resolved = self._resolve_company(prefix)
        if not resolved:
            return
        c = self._ensure_company(resolved)
        self._extract_quarterly_data(c, remainder)

    def _try_parse_capital_structure(self, line: str):
        """Parse 'COMPANY capital structure: D/E X.X. ... satisfaction Y.Y ...' lines."""
        m = re.match(r'^(.+?)\s+capital\s+structure\s*:\s*(.+)$', line, re.IGNORECASE)
        if not m:
            return
        prefix = m.group(1).strip()
        remainder = m.group(2).strip()
        resolved = self._resolve_company(prefix)
        if not resolved:
            return
        c = self._ensure_company(resolved)
        de_m = re.search(r'D/E\s*(\d+(?:\.\d+)?)', remainder)
        if de_m:
            c.de_ratio = float(de_m.group(1))
        sat_m = re.search(r'(\d+(?:\.\d+)?)\s+out\s+of\s+10', remainder)
        if sat_m:
            c.satisfaction = float(sat_m.group(1))

    def _try_parse_leverage_line(self, line: str):
        """Parse "COMPANY's leverage is ...: debt-to-equity sits at X.X. Employee sentiment is Y.Y/10" lines."""
        m = re.match(
            r"^(.+?)(?:'s)?\s+leverage\s+is\s+.+?debt-to-equity\s+sits\s+at\s+([\d.]+)\.\s*Employee\s+sentiment\s+is\s+([\d.]+)/10",
            line, re.IGNORECASE
        )
        if not m:
            return
        prefix = m.group(1).strip()
        resolved = self._resolve_company(prefix)
        if not resolved:
            return
        c = self._ensure_company(resolved)
        c.de_ratio = float(m.group(2))
        c.satisfaction = float(m.group(3))

    def _try_parse_at_company_line(self, line: str):
        """Parse 'At COMPANY, PERSON serves as TITLE. Headcount is N. ...' lines."""
        m = re.match(
            r'^At\s+(.+?),\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+serves\s+as\s+(.+?)\.\s*Headcount\s+is\s+(?:close\s+to|approximately|just\s+over|just\s+under|nearly|roughly|about|around)?\s*([\d,]+)',
            line, re.IGNORECASE
        )
        if not m:
            return
        company_text = m.group(1).strip()
        ceo_name = m.group(2).strip()
        emp_str = m.group(4).strip()
        resolved = self._resolve_company(company_text)
        if not resolved:
            return
        c = self._ensure_company(resolved)
        if not c.ceo_full_name:
            c.ceo_full_name = ceo_name
            c.ceo_last_name = ceo_name.split()[-1]
        emp = int(emp_str.replace(',', ''))
        if emp > 100:
            c.employees = emp
        # Parse founded/IPO from remainder
        fm = re.search(r'[Ff]ounded\s+(?:in\s+)?(\d{4})', line)
        if fm:
            c.founded_year = int(fm.group(1))
        ipo_m = re.search(r'went\s+public\s+in\s+(\d{4})', line, re.IGNORECASE)
        if ipo_m:
            c.is_public = True
            c.ipo_year = int(ipo_m.group(1))
        elif re.search(r'remains?\s+(?:privately\s+held|private)', line, re.IGNORECASE):
            c.is_public = False
            c.ipo_year = 0

    def _try_parse_for_company_revenue(self, line: str):
        """Parse 'For COMPANY, revenue came in at Q1: $XM (+Y%), ...' lines."""
        m = re.match(
            r'^For\s+(.+?),\s+revenue\s+came\s+in\s+at\s+(.+)$',
            line, re.IGNORECASE
        )
        if not m:
            return
        company_text = m.group(1).strip()
        remainder = m.group(2).strip()
        resolved = self._resolve_company(company_text)
        if not resolved:
            return
        c = self._ensure_company(resolved)
        self._extract_quarterly_data(c, remainder)

    # ----- Pass 3: Transcript lines -----

    def _pass_transcript(self):
        """Parse PANELIST/ANALYST/MODERATOR format lines.

        Uses a two-pass approach:
        Pass 1: Process metadata lines ("Let's jump to", "(SHORT) out of") to register
                 companies, DBA codes, and aliases. Then rebuild name lookup.
        Pass 2: Process data lines (staff count, revenue, D/E, etc.) using the
                 now-populated name lookup for DBA code resolution.
        """
        # Collect all transcript content lines
        transcript_contents = []
        for line in self.lines:
            line = line.strip()
            if not line:
                continue
            if is_counterfactual(line):
                continue
            m = re.match(r'^(?:PANELIST|ANALYST|MODERATOR|SPEAKER|HOST)\s*:\s*(.+)$', line, re.IGNORECASE)
            if not m:
                continue
            transcript_contents.append(m.group(1).strip())

        if not transcript_contents:
            return

        # Pre-register known company names so they can be resolved
        for kn in self.known_names:
            kn = kn.strip()
            if kn:
                self._ensure_company(kn)
                self._register_alias(kn, kn)
                words = kn.split()
                if len(words) > 1:
                    self._register_alias(words[0], kn)

        # Pass 1: Process metadata lines to register companies and DBA codes
        metadata_indices = set()
        for i, content in enumerate(transcript_contents):
            # "Let's jump to COMPANY from CITY, COUNTRY — SECTOR. People shorthand it as DBA."
            m = re.match(
                r"Let's jump to (.+?) from (\w+),\s*(\w+)\s*[—\-]\s*(.+?)\.\s*People shorthand it as (\w+)\.",
                content, re.IGNORECASE
            )
            if m:
                full_name = m.group(1).strip()
                city = m.group(2).strip()
                country = m.group(3).strip()
                sector = m.group(4).strip()
                dba = m.group(5).strip()
                c = self._ensure_company(full_name)
                c.hq_city = city
                c.hq_country = country
                c.sector = sector
                c.dba = dba
                self._register_alias(dba, full_name)
                self._register_alias(full_name.split()[0], full_name)
                metadata_indices.add(i)
                continue

            # "COMPANY (SHORT) out of CITY is in SECTOR."
            m = re.match(r"(.+?)\s*\((\w+)\)\s+out of\s+(\w+)\s+is in\s+(.+?)\.", content, re.IGNORECASE)
            if m:
                full_name = m.group(1).strip()
                short = m.group(2).strip()
                city = m.group(3).strip()
                sector = m.group(4).strip()
                c = self._ensure_company(full_name)
                c.short_name = short
                c.hq_city = city
                c.sector = sector
                self._register_alias(short, full_name)
                self._register_alias(full_name.split()[0], full_name)
                metadata_indices.add(i)
                continue

            # "COMPANY is a SECTOR company based in CITY, COUNTRY. People ... call it DBA ..."
            m = re.match(
                r'^(.+?)\s+is\s+a\s+(.+?)\s+company\s+based\s+in\s+(\w+),\s*(\w+)\.\s*People\s+.*call\s+it\s+(\w+)',
                content, re.IGNORECASE
            )
            if m:
                full_name = m.group(1).strip()
                sector = m.group(2).strip()
                city = m.group(3).strip()
                country = m.group(4).strip()
                dba = m.group(5).strip()
                c = self._ensure_company(full_name)
                c.sector = sector
                c.hq_city = city
                c.hq_country = country
                c.dba = dba
                self._register_alias(dba, full_name)
                self._register_alias(full_name.split()[0], full_name)
                metadata_indices.add(i)
                continue

            # "From CITY (COUNTRY), COMPANY operates in SECTOR. In internal notes ... as SHORT."
            m = re.match(
                r'^From\s+(\w+)\s*\((\w+)\),\s*(.+?)\s+operates\s+in\s+(.+?)\.\s*In\s+internal\s+notes\s+.*(?:as|called)\s+(\w+)',
                content, re.IGNORECASE
            )
            if m:
                city = m.group(1).strip()
                country = m.group(2).strip()
                full_name = m.group(3).strip()
                sector = m.group(4).strip()
                short_name = m.group(5).strip()
                c = self._ensure_company(full_name)
                c.hq_city = city
                c.hq_country = country
                c.sector = sector
                c.short_name = short_name
                self._register_alias(short_name, full_name)
                self._register_alias(full_name.split()[0], full_name)
                metadata_indices.add(i)
                continue

        # Rebuild name lookup with newly registered companies and DBA codes
        self._build_name_lookup()

        # Pass 2: Process data lines using the now-populated name lookup
        for i, content in enumerate(transcript_contents):
            if i in metadata_indices:
                continue
            self._parse_transcript_content(content)

    def _parse_transcript_content(self, content: str):
        """Parse the content portion of a transcript line (data lines only).

        Metadata lines ("Let's jump to" and "(SHORT) out of") are handled in
        _pass_transcript's first pass. This method handles data lines like:
        "GC staff count 77,793, CEO Emile Eriksen (Co-CEO). Founded 1983; went public in 1988."
        "At Xeno Net, Cedric Varga is president & ceo; 8,633 employees."
        "For NA, the book says opening quarter 1,683 million dollars / -15%, ..."
        "Apex Nexus D/E is 1.8, satisfaction 7.1/10."
        "Giga Cloud's quarters: Q1 1,297 million dollars (-1%), ..."
        """
        # Parse "For ABBREV, the book says..." revenue lines
        m = re.match(r"For\s+(\w+),\s+the book says\s+(.+)", content, re.IGNORECASE)
        if m:
            abbrev = m.group(1).strip()
            resolved = self._resolve_company(abbrev)
            if resolved:
                c = self._ensure_company(resolved)
                self._extract_quarterly_data(c, m.group(2))
                return

        # Parse "DBA tape: ..." revenue lines within transcript content
        m = re.match(r'^(\w+)\s+tape\s*:\s*(.+)$', content, re.IGNORECASE)
        if m:
            prefix = m.group(1).strip()
            remainder = m.group(2).strip()
            resolved = self._resolve_company(prefix)
            if resolved:
                c = self._ensure_company(resolved)
                self._extract_quarterly_data(c, remainder)
                return

        # Parse "COMPANY capital structure: D/E X.X. ... satisfaction Y.Y ..." lines
        m = re.match(r'^(.+?)\s+capital\s+structure\s*:\s*(.+)$', content, re.IGNORECASE)
        if m:
            prefix = m.group(1).strip()
            remainder = m.group(2).strip()
            resolved = self._resolve_company(prefix)
            if resolved:
                c = self._ensure_company(resolved)
                de_m = re.search(r'D/E\s*(\d+(?:\.\d+)?)', remainder)
                if de_m:
                    c.de_ratio = float(de_m.group(1))
                sat_m = re.search(r'(\d+(?:\.\d+)?)\s+out\s+of\s+10', remainder)
                if sat_m:
                    c.satisfaction = float(sat_m.group(1))
                return

        # Parse "COMPANY's leverage is ...: debt-to-equity sits at X.X. Employee sentiment is Y.Y/10"
        m = re.match(
            r"^(.+?)(?:'s)?\s+leverage\s+is\s+.+?debt-to-equity\s+sits\s+at\s+([\d.]+)\.\s*Employee\s+sentiment\s+is\s+([\d.]+)/10",
            content, re.IGNORECASE
        )
        if m:
            prefix = m.group(1).strip()
            resolved = self._resolve_company(prefix)
            if resolved:
                c = self._ensure_company(resolved)
                c.de_ratio = float(m.group(2))
                c.satisfaction = float(m.group(3))
                return

        # Parse "DBA management: Title Name. It employs N people. Founded YYYY; ..."
        m = re.match(r'^(\w+)\s+management\s*:\s*(.+)$', content, re.IGNORECASE)
        if m:
            prefix = m.group(1).strip()
            remainder = m.group(2).strip()
            resolved = self._resolve_company(prefix)
            if resolved:
                c = self._ensure_company(resolved)
                # Extract CEO name
                ceo_name = None
                for pat in [
                    r'(?:CEO|Co-CEO|Chief\s+Executive|Managing\s+Director|President\s*(?:&\s*CEO)?|Founder\s*(?:&\s*CEO)?)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
                    r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*(?:\((?:Co-)?CEO\))',
                ]:
                    cm = re.search(pat, remainder, re.IGNORECASE)
                    if cm:
                        ceo_name = cm.group(1).strip()
                        break
                if ceo_name and not c.ceo_full_name:
                    c.ceo_full_name = ceo_name
                    c.ceo_last_name = ceo_name.split()[-1]
                # Extract employee count
                emp_m = re.search(r'(?:employs?|It\s+employs?)\s+(?:nearly|approximately|roughly|about|close\s+to|just\s+over|just\s+under|around)?\s*([\d,]+)\s*(?:people|employees|staff|workers)', remainder, re.IGNORECASE)
                if emp_m:
                    emp = int(emp_m.group(1).replace(',', ''))
                    if emp > 100:
                        c.employees = emp
                # Extract founded year
                fm = re.search(r'[Ff]ounded\s+(?:in\s+)?(\d{4})', remainder)
                if fm:
                    c.founded_year = int(fm.group(1))
                # Extract IPO status
                ipo_m = re.search(r'went\s+public\s+in\s+(\d{4})', remainder, re.IGNORECASE)
                if ipo_m:
                    c.is_public = True
                    c.ipo_year = int(ipo_m.group(1))
                elif re.search(r'remains?\s+(?:privately\s+held|private)', remainder, re.IGNORECASE):
                    c.is_public = False
                    c.ipo_year = 0
                return

        # Try to identify which company this line is about
        company_name = None

        # Pattern: "At COMPANY, ..." or "For COMPANY, ..." or "COMPANY staff/D/E/..."
        for pat in [
            r'^(?:At|For|Per)\s+(.+?)\s*,',
            r'^(.+?)\s+(?:staff\s+count|D/E|debt|satisfaction|employees?)',
            r'^(.+?)\s+(?:revenue|financials|quarterly)',
            r"^(.+?)(?:'s)\s+quarters\b",
            r'^(.+?)\s+leverage\s+sits\s+at\b',
        ]:
            m = re.match(pat, content, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                resolved = self._resolve_company(candidate)
                if resolved:
                    company_name = resolved
                    break

        # Try: first 1-3 words as company identifier
        if not company_name:
            words = content.split()
            if words:
                for n in [3, 2, 1]:
                    if n <= len(words):
                        candidate = ' '.join(words[:n])
                        resolved = self._resolve_company(candidate)
                        if resolved:
                            company_name = resolved
                            break

        # Broader fallback — try each word and 2-word combo against name_lookup
        if not company_name:
            words = content.split()
            # Try 2-word combinations first (more specific)
            for i in range(len(words) - 1):
                candidate = words[i] + ' ' + words[i + 1]
                resolved = self._resolve_company(candidate)
                if resolved:
                    company_name = resolved
                    break
            # Then try individual words (skip common filler words)
            if not company_name:
                skip_words = {'the', 'a', 'an', 'is', 'at', 'for', 'per', 'and', 'or', 'of', 'in', 'to', 'it', 'its'}
                for w in words:
                    if w.lower() in skip_words:
                        continue
                    # Strip trailing punctuation for matching
                    clean = re.sub(r"[',;:.!?]+$", '', w)
                    if not clean:
                        continue
                    resolved = self._resolve_company(clean)
                    if resolved:
                        company_name = resolved
                        break

        if not company_name:
            return

        c = self._ensure_company(company_name)

        # Parse CEO info
        # "CEO Emile Eriksen" or "Emile Eriksen (Co-CEO)" or "Cedric Varga is president & ceo"
        ceo_patterns = [
            r'(?:CEO|Co-CEO|Chief\s+Executive|Managing\s+Director|President\s*(?:&\s*CEO)?|Founder\s*(?:&\s*CEO)?)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*(?:\((?:Co-)?CEO\)|is\s+(?:president\s*&\s*)?(?:co-)?ceo|is\s+(?:chief\s+executive|managing\s+director|founder\s*&\s*ceo|co-ceo))',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:\((?:Co-)?CEO\))',
        ]
        for pat in ceo_patterns:
            m = re.search(pat, content, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                if not c.ceo_full_name:
                    c.ceo_full_name = name
                    c.ceo_last_name = name.split()[-1]
                break

        # Parse employee count
        emp_patterns = [
            r'(?:staff\s+count)\s+(?:nearly|approximately|roughly|about|close\s+to|just\s+over|just\s+under|around)?\s*([\d,]+)',
            r'(?:employees?|employs?)\s+(?:nearly|approximately|roughly|about|close\s+to|just\s+over|just\s+under|around)?\s*([\d,]+)',
            r'([\d,]+)\s+(?:employees?|people|staff|workers)',
        ]
        for pat in emp_patterns:
            m = re.search(pat, content, re.IGNORECASE)
            if m:
                emp = int(m.group(1).replace(',', ''))
                if emp > 100:  # Sanity check
                    c.employees = emp
                break

        # Parse founded year
        fm = re.search(r'[Ff]ounded\s+(?:in\s+)?(\d{4})', content)
        if fm:
            c.founded_year = int(fm.group(1))

        # Parse IPO status
        ipo_m = re.search(r'went\s+public\s+in\s+(\d{4})', content, re.IGNORECASE)
        if ipo_m:
            c.is_public = True
            c.ipo_year = int(ipo_m.group(1))
        elif re.search(r'remains?\s+(?:privately\s+held|private)', content, re.IGNORECASE):
            c.is_public = False
            c.ipo_year = 0

        # Parse D/E ratio
        de_m = re.search(r'(?:D/E|[Dd]ebt[\s-]*to[\s-]*equity)\s*(?:is\s+|:?\s*)([\d.]+)', content)
        if de_m:
            c.de_ratio = float(de_m.group(1))
        else:
            # "leverage sits at 4.5 D/E" — number before D/E
            de_m2 = re.search(r'(?:leverage\s+sits\s+at|leverage\s+(?:is|at))\s+([\d.]+)\s*D/E', content, re.IGNORECASE)
            if de_m2:
                c.de_ratio = float(de_m2.group(1))

        # Parse satisfaction
        sat_m = re.search(r'[Ss]atisfaction\s*(?:is\s+|:?\s*)([\d.]+)(?:\s*/\s*10)?', content)
        if sat_m:
            c.satisfaction = float(sat_m.group(1))
        else:
            # "satisfaction reads 6.5"
            sat_m2 = re.search(r'satisfaction\s+reads\s+(\d+(?:\.\d+)?)', content, re.IGNORECASE)
            if sat_m2:
                c.satisfaction = float(sat_m2.group(1))

        # Parse revenue data if present
        has_quarter = re.search(
            r'(?:Q[1-4]|first\s+quarter|second\s+quarter|third\s+quarter|fourth\s+quarter|opening\s+quarter|closing\s+quarter|mid-spring|follow-on|pre-close|late-year|year-end|year\'s\s+first)',
            content, re.IGNORECASE
        )
        has_money = re.search(r'(?:\$|million|billion|M\b|B\b)', content, re.IGNORECASE)
        if has_quarter and has_money:
            self._extract_quarterly_data(c, content)


# =============================================================================
# QUESTION ENGINE
# =============================================================================

class QuestionEngine:
    """Deterministic question answering via regex classification + computation.

    Each question is matched against known patterns and dispatched to a
    handler that performs the appropriate filter/sort/aggregate operation.
    """

    def __init__(self, companies: Dict[str, CompanyData]):
        self.companies = companies
        self.handlers = self._build_handlers()

    def _build_handlers(self):
        """Build list of (compiled_regex, handler_function) pairs.

        Order matters: more specific patterns should come first.
        """
        return [
            # Type 12: Founded in decade + all positive growth (only one)
            (re.compile(r'(?:only\s+company|only\s+firm)\s+founded\s+in\s+the\s+(\d{4})s\s+with\s+positive\s+growth\s+in\s+every\s+quarter', re.I),
             self._founded_decade_all_positive),
            (re.compile(r'founded\s+in\s+the\s+(\d{4})s\s+(?:that\s+)?(?:had|with|showing)\s+positive\s+growth\s+(?:in\s+)?(?:every|all\s+four)\s+quarter', re.I),
             self._founded_decade_all_positive),

            # Type 11: Public + all positive growth + fewest employees
            (re.compile(r'public\s+compan(?:y|ies)\s+(?:that\s+)?(?:had|with|showing)\s+positive\s+growth\s+in\s+every\s+quarter.*(?:fewest|least|smallest)\s+(?:number\s+of\s+)?employees', re.I),
             self._public_all_positive_fewest_employees),
            (re.compile(r'(?:fewest|least|smallest)\s+(?:number\s+of\s+)?employees.*public.*positive\s+growth\s+in\s+every\s+quarter', re.I),
             self._public_all_positive_fewest_employees),

            # Type 19: D/E below X + satisfaction above Y + highest Q4 revenue
            (re.compile(r'D/E\s+(?:below|under|less\s+than)\s+([\d.]+)\s+and\s+satisfaction\s+(?:above|over|greater\s+than|exceeding)\s+([\d.]+).*(?:highest|largest|biggest|most)\s+(?:Q4|fourth[\s-]quarter|closing[\s-]quarter)\s+revenue', re.I),
             self._de_sat_highest_q4),
            (re.compile(r'(?:debt[\s-]*to[\s-]*equity|D/E)\s+(?:ratio\s+)?(?:below|under|less\s+than)\s+([\d.]+).*satisfaction\s+(?:above|over|greater\s+than|exceeding)\s+([\d.]+).*(?:highest|largest|biggest|most)\s+(?:Q4|fourth|closing)', re.I),
             self._de_sat_highest_q4),

            # Type 3: Founded before X + best closing quarter growth
            (re.compile(r'founded\s+(?:before|earlier\s+than|prior\s+to)\s+(\d{4}).*(?:strongest|highest|best|largest|biggest)\s+(?:growth|gain).*(?:closing|fourth|Q4|last)\s+quarter', re.I),
             self._founded_before_best_q4_growth),
            (re.compile(r'founded\s+(?:before|earlier\s+than|prior\s+to)\s+(\d{4}).*(?:Q4|fourth[\s-]quarter|closing[\s-]quarter)\s+growth', re.I),
             self._founded_before_best_q4_growth),

            # Type 1: Revenue volatility
            (re.compile(r'(?:highest|most|greatest|largest)\s+revenue\s+volatility', re.I),
             self._revenue_volatility),
            (re.compile(r'gap\s+between.*(?:best|highest)\s+and\s+(?:worst|lowest)\s+quarter', re.I),
             self._revenue_volatility),
            (re.compile(r'(?:largest|biggest|greatest)\s+(?:difference|gap|spread)\s+between.*(?:highest|best).*(?:lowest|worst)\s+(?:quarterly\s+)?revenue', re.I),
             self._revenue_volatility),

            # Type 2: Employees per million revenue
            (re.compile(r'(?:most|highest)\s+employees?\s+per\s+million', re.I),
             self._employees_per_revenue),
            (re.compile(r'highest\s+ratio\s+of\s+employees?\s+to.*revenue', re.I),
             self._employees_per_revenue),
            (re.compile(r'employees?\s+(?:per|divided\s+by|to).*(?:total\s+)?(?:annual\s+)?revenue', re.I),
             self._employees_per_revenue),

            # Type 4: Largest Q1→Q4 revenue increase
            (re.compile(r'(?:largest|biggest|greatest|highest)\s+(?:revenue\s+)?(?:increase|jump|gain|rise).*(?:Q1|first[\s-]quarter|opening[\s-]quarter).*(?:Q4|fourth[\s-]quarter|closing[\s-]quarter)', re.I),
             self._largest_q1_to_q4_increase),
            (re.compile(r'(?:jump|increase|gain|rise).*(?:first|Q1|opening).*(?:fourth|Q4|closing).*(?:biggest|largest|greatest|highest)', re.I),
             self._largest_q1_to_q4_increase),
            (re.compile(r'(?:Q4|fourth[\s-]quarter|closing[\s-]quarter)\s+(?:revenue\s+)?minus\s+(?:Q1|first[\s-]quarter|opening[\s-]quarter)', re.I),
             self._largest_q1_to_q4_increase),

            # Type 5: Public + lowest D/E
            (re.compile(r'public\s+compan(?:y|ies).*(?:lowest|smallest|least)\s+(?:debt[\s-]*to[\s-]*equity|D/E)', re.I),
             self._public_lowest_de),
            (re.compile(r'(?:lowest|smallest|least)\s+(?:debt[\s-]*to[\s-]*equity|D/E).*public', re.I),
             self._public_lowest_de),

            # Type 6: Most recent IPO
            (re.compile(r'(?:most\s+recent|latest|newest)\s+IPO', re.I),
             self._most_recent_ipo),
            (re.compile(r'IPO.{0,5}d?\s+most\s+recently', re.I),
             self._most_recent_ipo),
            (re.compile(r'went\s+public\s+most\s+recently', re.I),
             self._most_recent_ipo),

            # Type 7: Over X employees + least revenue
            (re.compile(r'(?:over|more\s+than|exceeding|above)\s+([\d,]+)\s+employees?.*(?:least|lowest|smallest|minimum)\s+(?:total\s+)?revenue', re.I),
             self._over_employees_least_revenue),
            (re.compile(r'(?:least|lowest|smallest|minimum)\s+(?:total\s+)?revenue.*(?:over|more\s+than|exceeding|above)\s+([\d,]+)\s+employees?', re.I),
             self._over_employees_least_revenue),

            # Type 17: Over $1B revenue + lowest satisfaction
            (re.compile(r'(?:over|more\s+than|exceeding|above)\s+\$?1\s*(?:billion|B)\s+(?:in\s+)?(?:total\s+)?(?:annual\s+)?revenue.*(?:lowest|worst|least|smallest)\s+satisfaction', re.I),
             self._over_1b_lowest_satisfaction),
            (re.compile(r'(?:lowest|worst|least|smallest)\s+satisfaction.*(?:over|more\s+than|exceeding|above)\s+\$?1\s*(?:billion|B)', re.I),
             self._over_1b_lowest_satisfaction),

            # Type 10/20: Q3 decline + highest D/E
            (re.compile(r'(?:decline|drop|decrease|negative\s+growth|loss)\s+in\s+(?:Q3|third[\s-]quarter|pre-close).*(?:highest|largest|biggest|greatest)\s+(?:D/E|debt[\s-]*to[\s-]*equity)', re.I),
             self._q3_decline_highest_de),
            (re.compile(r'(?:highest|largest|biggest|greatest)\s+(?:D/E|debt[\s-]*to[\s-]*equity).*(?:decline|drop|decrease|negative)\s+(?:in\s+)?(?:Q3|third[\s-]quarter)', re.I),
             self._q3_decline_highest_de),

            # Type 9: Highest total revenue
            (re.compile(r'(?:highest|largest|biggest|greatest|most)\s+(?:total\s+)?(?:annual\s+)?revenue.*(?:all\s+four|entire\s+year|full[\s-]year|year)', re.I),
             self._highest_total_revenue),
            (re.compile(r'full[\s-]year\s+sum.*(?:top|highest|largest|biggest)', re.I),
             self._highest_total_revenue),
            (re.compile(r'(?:comes?\s+out\s+on\s+top|produced\s+the\s+(?:largest|most|highest)\s+total\s+revenue)', re.I),
             self._highest_total_revenue),

            # Type 15: Highest average growth
            (re.compile(r'(?:highest|largest|biggest|greatest|best)\s+(?:average|mean)\s+(?:quarterly\s+)?growth', re.I),
             self._highest_avg_growth),
            (re.compile(r'[Aa]veraging\s+the\s+growth\s+rates.*(?:top|highest|comes?\s+out\s+on\s+top)', re.I),
             self._highest_avg_growth),

            # Type 16: Second oldest
            (re.compile(r'second[\s-](?:oldest|earliest).*(?:founding|founded)', re.I),
             self._second_oldest),
            (re.compile(r'(?:founding|founded).*second[\s-](?:oldest|earliest)', re.I),
             self._second_oldest),
            (re.compile(r'second[\s-]earliest\s+founding', re.I),
             self._second_oldest),

            # Type 14: Private + best satisfaction
            (re.compile(r'(?:private|privately\s+held).*(?:best|highest|top)\s+satisfaction', re.I),
             self._private_best_satisfaction),
            (re.compile(r'(?:best|highest|top)\s+satisfaction.*(?:private|privately\s+held)', re.I),
             self._private_best_satisfaction),
            (re.compile(r'remain(?:ed|s)?\s+private.*(?:best|highest|top)\s+satisfaction', re.I),
             self._private_best_satisfaction),

            # Type 21: Private + most employees
            (re.compile(r'(?:private|privately\s+held).*(?:most|largest|biggest|greatest|highest)\s+(?:number\s+of\s+)?employees', re.I),
             self._private_most_employees),
            (re.compile(r'(?:most|largest|biggest|greatest|highest)\s+(?:number\s+of\s+)?employees.*(?:private|privately\s+held)', re.I),
             self._private_most_employees),

            # Type 8/23: Sector + founded earliest
            (re.compile(r'(?:in\s+the\s+)?(\w+)\s+sector.*(?:founded|oldest)\s+(?:earliest|first|oldest)', re.I),
             self._sector_founded_earliest),
            (re.compile(r'(?:oldest|earliest).*(?:in\s+the\s+)?(\w+)\s+sector', re.I),
             self._sector_founded_earliest),

            # Type 13: City + highest Q2 revenue
            (re.compile(r'headquartered\s+in\s+(\w+).*(?:highest|largest|biggest)\s+(?:Q2|second[\s-]quarter|mid-spring)\s+revenue', re.I),
             self._city_highest_q2),
            (re.compile(r'(?:highest|largest|biggest)\s+(?:Q2|second[\s-]quarter)\s+revenue.*headquartered\s+in\s+(\w+)', re.I),
             self._city_highest_q2),

            # Type 18: City/Country + most employees
            (re.compile(r'headquartered\s+in\s+(\w+).*(?:largest|most|biggest|greatest|highest)\s+(?:employee\s+count|number\s+of\s+employees|employees)', re.I),
             self._location_most_employees),
            (re.compile(r'(?:in|from)\s+(\w+).*(?:HQ|headquartered).*(?:most|largest|biggest)\s+(?:employees|people)', re.I),
             self._location_most_employees),
            (re.compile(r'[Ii]n\s+(\w+)\s*,\s*which\s+HQ.*(?:most|largest|biggest)\s+(?:employees|people)', re.I),
             self._location_most_employees),
        ]

    def answer(self, question: str) -> Optional[str]:
        """Answer a question by matching it against known patterns.

        Returns the company name that answers the question, or None.
        """
        q = question.strip()
        dbg(f"\n--- QUESTION: {q}")

        for pattern, handler in self.handlers:
            m = pattern.search(q)
            if m:
                result = handler(q, m)
                if result:
                    dbg(f"    ANSWER: {result} (handler: {handler.__name__})")
                    return result

        # Fallback: try broader pattern matching
        result = self._fallback_answer(q)
        if result:
            dbg(f"    ANSWER (fallback): {result}")
            return result

        dbg(f"    NO ANSWER FOUND")
        return None


    # ----- Handler implementations -----

    def _revenue_volatility(self, q: str, m: re.Match) -> Optional[str]:
        """Type 1: Which company had the highest revenue volatility?"""
        best_name, best_vol = None, -1
        for name, c in self.companies.items():
            if not c.has_revenue():
                continue
            vol = c.revenue_volatility()
            dbg(f"      {name}: volatility={vol} (rev={c.revenue})")
            if vol > best_vol:
                best_vol = vol
                best_name = name
        return best_name

    def _employees_per_revenue(self, q: str, m: re.Match) -> Optional[str]:
        """Type 2: Which firm has the most employees per million $ of annual revenue?"""
        best_name, best_ratio = None, -1.0
        for name, c in self.companies.items():
            total = c.total_revenue()
            if total <= 0 or c.employees <= 0:
                continue
            ratio = c.employees / total
            dbg(f"      {name}: emp/rev={ratio:.4f} (emp={c.employees}, rev={total})")
            if ratio > best_ratio:
                best_ratio = ratio
                best_name = name
        return best_name

    def _founded_before_best_q4_growth(self, q: str, m: re.Match) -> Optional[str]:
        """Type 3: Among companies founded before X, which had best Q4 growth?"""
        year_threshold = int(m.group(1))
        best_name, best_growth = None, -999.0
        for name, c in self.companies.items():
            if c.founded_year <= 0 or c.founded_year >= year_threshold:
                continue
            g = c.growth[3]
            if g is None:
                continue
            dbg(f"      {name}: founded={c.founded_year}, Q4_growth={g}")
            if g > best_growth:
                best_growth = g
                best_name = name
        return best_name

    def _largest_q1_to_q4_increase(self, q: str, m: re.Match) -> Optional[str]:
        """Type 4: Which company had the largest Q4-Q1 revenue increase?"""
        best_name, best_diff = None, -999999
        for name, c in self.companies.items():
            if not c.has_revenue():
                continue
            diff = c.revenue[3] - c.revenue[0]
            dbg(f"      {name}: Q4-Q1={diff} (Q1={c.revenue[0]}, Q4={c.revenue[3]})")
            if diff > best_diff:
                best_diff = diff
                best_name = name
        return best_name

    def _public_lowest_de(self, q: str, m: re.Match) -> Optional[str]:
        """Type 5: Among public companies, which has the lowest D/E ratio?"""
        best_name, best_de = None, 999999.0
        for name, c in self.companies.items():
            if not c.is_public:
                continue
            if c.de_ratio <= 0:
                continue
            dbg(f"      {name}: D/E={c.de_ratio}, public={c.is_public}")
            if c.de_ratio < best_de:
                best_de = c.de_ratio
                best_name = name
        return best_name

    def _most_recent_ipo(self, q: str, m: re.Match) -> Optional[str]:
        """Type 6: Which publicly traded company IPO'd most recently?"""
        best_name, best_year = None, 0
        for name, c in self.companies.items():
            if not c.is_public or c.ipo_year <= 0:
                continue
            dbg(f"      {name}: ipo_year={c.ipo_year}")
            if c.ipo_year > best_year:
                best_year = c.ipo_year
                best_name = name
        return best_name

    def _over_employees_least_revenue(self, q: str, m: re.Match) -> Optional[str]:
        """Type 7: Which company with over X employees had the least total revenue?"""
        threshold = int(m.group(1).replace(',', ''))
        best_name, best_rev = None, 999999999
        for name, c in self.companies.items():
            if c.employees <= threshold:
                continue
            total = c.total_revenue()
            if total <= 0:
                continue
            dbg(f"      {name}: emp={c.employees}, total_rev={total}")
            if total < best_rev:
                best_rev = total
                best_name = name
        return best_name

    def _sector_founded_earliest(self, q: str, m: re.Match) -> Optional[str]:
        """Type 8: In the X sector, which company was founded earliest?"""
        sector_kw = m.group(1).strip().lower()
        best_name, best_year = None, 999999
        for name, c in self.companies.items():
            if not c.sector:
                continue
            # Match sector keyword against first word of sector or full sector
            sector_lower = c.sector.lower()
            sector_first_word = sector_lower.split()[0] if sector_lower.split() else ""
            if sector_kw != sector_first_word and sector_kw not in sector_lower:
                continue
            if c.founded_year <= 0:
                continue
            dbg(f"      {name}: sector={c.sector}, founded={c.founded_year}")
            if c.founded_year < best_year:
                best_year = c.founded_year
                best_name = name
        return best_name

    def _highest_total_revenue(self, q: str, m: re.Match) -> Optional[str]:
        """Type 9: Which company had the highest total revenue?"""
        best_name, best_rev = None, -1
        for name, c in self.companies.items():
            total = c.total_revenue()
            dbg(f"      {name}: total_rev={total}")
            if total > best_rev:
                best_rev = total
                best_name = name
        return best_name

    def _q3_decline_highest_de(self, q: str, m: re.Match) -> Optional[str]:
        """Type 10/20: Which company with Q3 decline has the highest D/E?"""
        best_name, best_de = None, -1.0
        for name, c in self.companies.items():
            g3 = c.growth[2]
            if g3 is None or g3 >= 0:
                continue
            if c.de_ratio <= 0:
                continue
            dbg(f"      {name}: Q3_growth={g3}, D/E={c.de_ratio}")
            if c.de_ratio > best_de:
                best_de = c.de_ratio
                best_name = name
        return best_name

    def _public_all_positive_fewest_employees(self, q: str, m: re.Match) -> Optional[str]:
        """Type 11: Among public companies with all positive growth, fewest employees."""
        best_name, best_emp = None, 999999999
        for name, c in self.companies.items():
            if not c.is_public:
                continue
            if not c.all_growth_positive():
                continue
            if c.employees <= 0:
                continue
            dbg(f"      {name}: public, all_positive, emp={c.employees}")
            if c.employees < best_emp:
                best_emp = c.employees
                best_name = name
        return best_name

    def _founded_decade_all_positive(self, q: str, m: re.Match) -> Optional[str]:
        """Type 12: The only company founded in the XXXXs with all positive growth."""
        decade = int(m.group(1))
        matches = []
        for name, c in self.companies.items():
            if c.founded_year < decade or c.founded_year >= decade + 10:
                continue
            if not c.all_growth_positive():
                continue
            dbg(f"      {name}: founded={c.founded_year}, all_positive=True")
            matches.append(name)
        if len(matches) == 1:
            return matches[0]
        elif matches:
            dbg(f"      WARNING: Expected 1 match, got {len(matches)}: {matches}")
            return matches[0]
        return None

    def _city_highest_q2(self, q: str, m: re.Match) -> Optional[str]:
        """Type 13: Among companies in city X, which had highest Q2 revenue?"""
        city = m.group(1).strip().lower()
        best_name, best_q2 = None, -1
        for name, c in self.companies.items():
            if c.hq_city.lower() != city:
                continue
            q2 = c.revenue[1]
            dbg(f"      {name}: city={c.hq_city}, Q2={q2}")
            if q2 > best_q2:
                best_q2 = q2
                best_name = name
        return best_name

    def _private_best_satisfaction(self, q: str, m: re.Match) -> Optional[str]:
        """Type 14: Among private firms, which has the best satisfaction?"""
        best_name, best_sat = None, -1.0
        for name, c in self.companies.items():
            if c.is_public:
                continue
            if c.satisfaction <= 0:
                continue
            dbg(f"      {name}: private, satisfaction={c.satisfaction}")
            if c.satisfaction > best_sat:
                best_sat = c.satisfaction
                best_name = name
        return best_name

    def _highest_avg_growth(self, q: str, m: re.Match) -> Optional[str]:
        """Type 15: Which company had the highest average quarterly growth?"""
        best_name, best_avg = None, -999.0
        for name, c in self.companies.items():
            avg = c.avg_growth()
            valid = [g for g in c.growth if g is not None]
            if len(valid) < 4:
                continue
            dbg(f"      {name}: avg_growth={avg:.2f}")
            if avg > best_avg:
                best_avg = avg
                best_name = name
        return best_name

    def _second_oldest(self, q: str, m: re.Match) -> Optional[str]:
        """Type 16: Which firm is the second-oldest by founding date?"""
        founded = [(name, c.founded_year) for name, c in self.companies.items() if c.founded_year > 0]
        if len(founded) < 2:
            return None
        # Sort by year ascending, then alphabetically for ties
        founded.sort(key=lambda x: (x[1], x[0]))
        dbg(f"      Founded order: {[(n, y) for n, y in founded[:5]]}")
        return founded[1][0]

    def _over_1b_lowest_satisfaction(self, q: str, m: re.Match) -> Optional[str]:
        """Type 17: Among companies with >$1B total revenue, lowest satisfaction."""
        best_name, best_sat = None, 999.0
        for name, c in self.companies.items():
            total = c.total_revenue()
            if total <= 1000:
                continue
            if c.satisfaction <= 0:
                continue
            dbg(f"      {name}: total_rev={total}, satisfaction={c.satisfaction}")
            if c.satisfaction < best_sat:
                best_sat = c.satisfaction
                best_name = name
        return best_name

    def _location_most_employees(self, q: str, m: re.Match) -> Optional[str]:
        """Type 18: Which company in city/country X has the most employees?"""
        location = m.group(1).strip().lower()
        best_name, best_emp = None, -1
        for name, c in self.companies.items():
            if c.hq_city.lower() != location and c.hq_country.lower() != location:
                continue
            dbg(f"      {name}: city={c.hq_city}, country={c.hq_country}, emp={c.employees}")
            if c.employees > best_emp:
                best_emp = c.employees
                best_name = name
        return best_name

    def _de_sat_highest_q4(self, q: str, m: re.Match) -> Optional[str]:
        """Type 19: D/E below X, satisfaction above Y, highest Q4 revenue."""
        de_threshold = float(m.group(1))
        sat_threshold = float(m.group(2))
        best_name, best_q4 = None, -1
        for name, c in self.companies.items():
            if c.de_ratio <= 0 or c.de_ratio >= de_threshold:
                continue
            if c.satisfaction <= sat_threshold:
                continue
            q4 = c.revenue[3]
            dbg(f"      {name}: D/E={c.de_ratio}, sat={c.satisfaction}, Q4={q4}")
            if q4 > best_q4:
                best_q4 = q4
                best_name = name
        return best_name

    def _private_most_employees(self, q: str, m: re.Match) -> Optional[str]:
        """Type 21: Among privately held companies, which has the most employees?"""
        best_name, best_emp = None, -1
        for name, c in self.companies.items():
            if c.is_public:
                continue
            if c.employees <= 0:
                continue
            dbg(f"      {name}: private, emp={c.employees}")
            if c.employees > best_emp:
                best_emp = c.employees
                best_name = name
        return best_name


    def _fallback_answer(self, q: str) -> Optional[str]:
        """Fallback handler for questions that don't match specific patterns.

        Uses keyword extraction to determine the question type.
        """
        ql = q.lower()

        # Revenue volatility fallback
        if 'volatility' in ql and 'revenue' in ql:
            return self._revenue_volatility(q, None)

        # Employees per revenue fallback
        if 'employees' in ql and 'per' in ql and 'revenue' in ql:
            return self._employees_per_revenue(q, None)

        # Total revenue fallback
        if ('total' in ql or 'full-year' in ql or 'full year' in ql or 'all four' in ql) and 'revenue' in ql:
            if 'highest' in ql or 'largest' in ql or 'most' in ql or 'top' in ql or 'biggest' in ql:
                return self._highest_total_revenue(q, None)

        # Average growth fallback
        if 'average' in ql and 'growth' in ql:
            return self._highest_avg_growth(q, None)

        # Second oldest fallback
        if 'second' in ql and ('oldest' in ql or 'earliest' in ql):
            return self._second_oldest(q, None)

        # Most recent IPO fallback
        if 'ipo' in ql and ('recent' in ql or 'latest' in ql):
            return self._most_recent_ipo(q, None)

        # Public + lowest D/E fallback
        if 'public' in ql and ('lowest' in ql or 'smallest' in ql) and ('d/e' in ql or 'debt' in ql):
            return self._public_lowest_de(q, None)

        # Private + satisfaction fallback
        if ('private' in ql or 'privately' in ql) and 'satisfaction' in ql:
            if 'best' in ql or 'highest' in ql or 'top' in ql:
                return self._private_best_satisfaction(q, None)

        # Private + employees fallback
        if ('private' in ql or 'privately' in ql) and 'employees' in ql:
            if 'most' in ql or 'largest' in ql or 'biggest' in ql:
                return self._private_most_employees(q, None)

        # Q3 decline + D/E fallback
        if ('q3' in ql or 'third quarter' in ql) and ('decline' in ql or 'negative' in ql or 'drop' in ql) and ('d/e' in ql or 'debt' in ql):
            return self._q3_decline_highest_de(q, None)

        # Over X employees + least revenue fallback
        emp_m = re.search(r'(?:over|more\s+than)\s+([\d,]+)\s+employees?', ql)
        if emp_m and ('least' in ql or 'lowest' in ql or 'smallest' in ql) and 'revenue' in ql:
            return self._over_employees_least_revenue(q, emp_m)

        # Over $1B + lowest satisfaction fallback
        if ('billion' in ql or '$1b' in ql or '$1 b' in ql) and 'satisfaction' in ql:
            return self._over_1b_lowest_satisfaction(q, None)

        # Sector + founded earliest fallback
        sector_m = re.search(r'(?:in\s+the\s+)?(\w+)\s+sector', ql)
        if sector_m and ('oldest' in ql or 'earliest' in ql or 'founded' in ql):
            return self._sector_founded_earliest(q, sector_m)

        # City + Q2 revenue fallback
        city_m = re.search(r'headquartered\s+in\s+(\w+)', ql)
        if city_m and ('q2' in ql or 'second quarter' in ql) and 'revenue' in ql:
            return self._city_highest_q2(q, city_m)

        # Location + employees fallback
        loc_m = re.search(r'(?:headquartered|HQ|based)\s+in\s+(\w+)', ql)
        if not loc_m:
            loc_m = re.search(r'[Ii]n\s+(\w+)\s*,?\s*which', ql)
        if loc_m and ('employees' in ql or 'employs' in ql):
            return self._location_most_employees(q, loc_m)

        # Founded before + Q4 growth fallback
        before_m = re.search(r'founded\s+(?:before|earlier\s+than|prior\s+to)\s+(\d{4})', ql)
        if before_m and ('q4' in ql or 'fourth' in ql or 'closing' in ql) and 'growth' in ql:
            return self._founded_before_best_q4_growth(q, before_m)

        # Q1 to Q4 increase fallback
        if ('q1' in ql or 'first quarter' in ql or 'opening' in ql) and ('q4' in ql or 'fourth quarter' in ql or 'closing' in ql):
            if 'increase' in ql or 'jump' in ql or 'gain' in ql or 'rise' in ql or 'biggest' in ql or 'largest' in ql:
                return self._largest_q1_to_q4_increase(q, None)

        # Founded decade + all positive fallback
        decade_m = re.search(r'founded\s+in\s+the\s+(\d{4})s', ql)
        if decade_m and ('positive' in ql and 'growth' in ql):
            return self._founded_decade_all_positive(q, decade_m)

        # Public + all positive + fewest employees fallback
        if 'public' in ql and 'positive' in ql and 'growth' in ql and ('fewest' in ql or 'least' in ql) and 'employees' in ql:
            return self._public_all_positive_fewest_employees(q, None)

        # D/E + satisfaction compound filter fallback
        de_sat_m = re.search(r'D/E\s+(?:below|under)\s+([\d.]+).*satisfaction\s+(?:above|over)\s+([\d.]+)', ql, re.I)
        if de_sat_m:
            return self._de_sat_highest_q4(q, de_sat_m)

        return None



# =============================================================================
# CONSTRAINT ENGINE
# =============================================================================

class ConstraintEngine:
    """Parse and compute all constraint values from the challenge constraints.

    Extracts required elements (city, CEO name, country, prime, equation, acrostic)
    and forbidden letters from constraint text.
    """

    def __init__(self, constraints: List[str], questions: List[str],
                 answers: List[Optional[str]], companies: Dict[str, CompanyData]):
        self.constraints = constraints
        self.questions = questions
        self.answers = answers
        self.companies = companies

    def compute_all(self) -> dict:
        """Compute all constraint requirements.

        Returns dict with keys:
            word_count: int
            required_elements: list of strings that must appear in artifact
            acrostic: str of 8 characters (or empty)
            forbidden_letter: str (single char, or empty)
            equation: str like "42+57=99" (or empty)
        """
        result = {
            'word_count': 16,  # default
            'required_elements': [],
            'acrostic': '',
            'forbidden_letter': '',
            'equation': '',
        }

        for i, constraint in enumerate(self.constraints):
            ct = constraint.strip()
            dbg(f"\n  CONSTRAINT {i}: {ct}")
            self._parse_constraint(ct, result)

        dbg(f"\n=== CONSTRAINT SUMMARY ===")
        dbg(f"  word_count={result['word_count']}")
        dbg(f"  required_elements={result['required_elements']}")
        dbg(f"  acrostic='{result['acrostic']}'")
        dbg(f"  forbidden_letter='{result['forbidden_letter']}'")
        dbg(f"  equation='{result['equation']}'")

        return result

    def _get_answer_company(self, q_idx: int) -> Optional[CompanyData]:
        """Get the CompanyData for the answer to question q_idx (0-based)."""
        if q_idx < 0 or q_idx >= len(self.answers):
            return None
        ans = self.answers[q_idx]
        if not ans:
            return None
        return self.companies.get(ans)

    def _extract_question_ref(self, text: str) -> Optional[int]:
        """Extract a question reference number from constraint text.

        "Question 3" → 2 (0-based index)
        "Q3" → 2
        """
        m = re.search(r'[Qq]uestion\s+(\d+)', text)
        if m:
            return int(m.group(1)) - 1
        m = re.search(r'\bQ(\d+)\b', text)
        if m:
            return int(m.group(1)) - 1
        return None

    def _extract_all_question_refs(self, text: str) -> List[int]:
        """Extract all question reference numbers from constraint text."""
        refs = []
        for m in re.finditer(r'[Qq]uestion\s+(\d+)', text):
            refs.append(int(m.group(1)) - 1)
        return refs

    def _parse_constraint(self, text: str, result: dict):
        """Parse a single constraint and update result dict."""
        tl = text.lower()

        # C0: Word count
        wc_m = re.search(r'(?:write\s+)?(?:exactly|precisely)\s+(\d+)\s+words?', text, re.I)
        if wc_m:
            result['word_count'] = int(wc_m.group(1))
            dbg(f"    → word_count={result['word_count']}")
            return

        # C7: Forbidden letter
        fl_m = re.search(r'must\s+not\s+contain\s+the\s+letter\s+["\']?([a-zA-Z])["\']?', text, re.I)
        if fl_m:
            result['forbidden_letter'] = fl_m.group(1).lower()
            dbg(f"    → forbidden_letter='{result['forbidden_letter']}'")
            return

        # C6: Acrostic
        acr_m = re.search(r'[Aa]crostic.*first\s+(\d+)\s+(?:letters|chars)', text)
        if not acr_m:
            acr_m = re.search(r'[Aa]crostic.*first\s+letters\s+of\s+the\s+first\s+(\d+)\s+words', text)
        if not acr_m:
            acr_m = re.search(r'[Aa]crostic', text, re.I)

        if 'acrostic' in tl:
            # Extract the number of acrostic letters (usually 8)
            num_m = re.search(r'first\s+(\d+)\s+(?:letters|words)', text)
            acr_len = int(num_m.group(1)) if num_m else 8

            # Extract question references for initials
            refs = self._extract_all_question_refs(text)
            if not refs:
                # Try initials(QN) pattern — these are always question references
                initials_refs = re.findall(r'initials\(Q(\d+)\)', text)
                if initials_refs:
                    refs = [int(r) - 1 for r in initials_refs]
                else:
                    # Try Q1, Q2, etc. format
                    for qm in re.finditer(r'\bQ(\d+)\b', text):
                        val = int(qm.group(1))
                        refs.append(val - 1)
                    # Filter out quarter labels if text has revenue context
                    if any(kw in text.lower() for kw in ['revenue', 'quarter', 'mod']):
                        refs = [r for r in refs if r >= 4]  # Q5+ are question refs, Q1-Q4 are quarters

            # Also try "QA", "QB" etc. pattern — these refer to question answers
            qa_refs = re.findall(r'Q([A-Z])', text)
            if qa_refs:
                refs = []
                for ref in qa_refs:
                    # QA = Question 1, QB = Question 2, etc.
                    idx = ord(ref) - ord('A')
                    refs.append(idx)

            # Build acrostic from company initials
            initials_str = ''
            for ref in refs:
                ans_company = self._get_answer_company(ref)
                if ans_company:
                    initials_str += get_initials(ans_company.name)
                    dbg(f"    → Q{ref+1} answer '{ans_company.name}' → initials '{get_initials(ans_company.name)}'")

            result['acrostic'] = initials_str[:acr_len].upper()
            dbg(f"    → acrostic='{result['acrostic']}' (from refs {[r+1 for r in refs]})")
            return

        # C1: HQ City
        city_m = re.search(r'(?:headquarters?\s+city|HQ\s+city|city\s+of\s+(?:the\s+)?headquarters?)', text, re.I)
        if city_m:
            ref = self._extract_question_ref(text)
            if ref is not None:
                c = self._get_answer_company(ref)
                if c and c.hq_city:
                    result['required_elements'].append(c.hq_city)
                    dbg(f"    → required city: '{c.hq_city}' (from Q{ref+1})")
            return

        # C3: HQ Country
        country_m = re.search(r'(?:headquarters?\s+country|HQ\s+country|country\s+of\s+(?:the\s+)?headquarters?)', text, re.I)
        if country_m:
            ref = self._extract_question_ref(text)
            if ref is not None:
                c = self._get_answer_company(ref)
                if c and c.hq_country:
                    result['required_elements'].append(c.hq_country)
                    dbg(f"    → required country: '{c.hq_country}' (from Q{ref+1})")
            return

        # C2: CEO Last Name
        ceo_m = re.search(r"(?:CEO'?s?\s+last\s+name|last\s+name\s+of\s+(?:the\s+)?CEO)", text, re.I)
        if ceo_m:
            ref = self._extract_question_ref(text)
            if ref is not None:
                c = self._get_answer_company(ref)
                if c and c.ceo_last_name:
                    result['required_elements'].append(c.ceo_last_name)
                    dbg(f"    → required CEO last name: '{c.ceo_last_name}' (from Q{ref+1})")
            return

        # C4: Prime number
        prime_m = re.search(r'prime', text, re.I)
        if prime_m and 'employees' in tl:
            ref = self._extract_question_ref(text)
            if ref is not None:
                c = self._get_answer_company(ref)
                if c:
                    val = (c.employees % 100) + 11
                    p = next_prime(val)
                    result['required_elements'].append(str(p))
                    dbg(f"    → prime: employees={c.employees}, mod100={c.employees%100}, +11={val}, nextPrime={p}")
            return

        # C5: Equation
        eq_m = re.search(r'equation', text, re.I)
        if eq_m or ('A+B=C' in text) or ('A + B = C' in text):
            # Extract two question references
            refs = self._extract_all_question_refs(text)
            if len(refs) >= 2:
                c_a = self._get_answer_company(refs[0])
                c_b = self._get_answer_company(refs[1])
                if c_a and c_b:
                    a_val = (c_a.revenue[0] % 90) + 10  # Q1 revenue mod 90 + 10
                    b_val = (c_b.revenue[3] % 90) + 10  # Q4 revenue mod 90 + 10
                    c_val = a_val + b_val
                    eq_str = f"{a_val}+{b_val}={c_val}"
                    result['equation'] = eq_str
                    result['required_elements'].append(eq_str)
                    dbg(f"    → equation: A={a_val} (Q1 of {c_a.name}={c_a.revenue[0]}), "
                        f"B={b_val} (Q4 of {c_b.name}={c_b.revenue[3]}), C={c_val}")
            elif len(refs) == 1:
                # Both A and B from same question's answer
                c = self._get_answer_company(refs[0])
                if c:
                    a_val = (c.revenue[0] % 90) + 10
                    b_val = (c.revenue[3] % 90) + 10
                    c_val = a_val + b_val
                    eq_str = f"{a_val}+{b_val}={c_val}"
                    result['equation'] = eq_str
                    result['required_elements'].append(eq_str)
            return

        # C8: Tip (no action needed, just informational)
        if 'tip' in tl or 'avoid extra punctuation' in tl:
            dbg(f"    → tip (informational only)")
            return

        # Generic: "Must include" with question reference
        include_m = re.search(r'[Mm]ust\s+include\s+(?:the\s+)?(.+?)(?:\s+of\s+(?:the\s+)?(?:company\s+that\s+)?answers?\s+)?[Qq]uestion\s+(\d+)', text)
        if include_m:
            what = include_m.group(1).strip().lower()
            ref = int(include_m.group(2)) - 1
            c = self._get_answer_company(ref)
            if c:
                if 'city' in what or 'headquarters' in what:
                    if c.hq_city:
                        result['required_elements'].append(c.hq_city)
                        dbg(f"    → required (generic city): '{c.hq_city}'")
                elif 'country' in what:
                    if c.hq_country:
                        result['required_elements'].append(c.hq_country)
                        dbg(f"    → required (generic country): '{c.hq_country}'")
                elif 'ceo' in what or 'last name' in what:
                    if c.ceo_last_name:
                        result['required_elements'].append(c.ceo_last_name)
                        dbg(f"    → required (generic CEO): '{c.ceo_last_name}'")
            return

        dbg(f"    → UNRECOGNIZED CONSTRAINT")


# =============================================================================
# ARTIFACT GENERATOR
# =============================================================================

# Word bank organized by first letter, avoiding common forbidden letters
# Each entry: letter → list of common words starting with that letter
WORD_BANK = {
    'a': ['also', 'area', 'art', 'aim', 'arc', 'arm', 'ask', 'atom', 'auto', 'axis'],
    'b': ['bold', 'born', 'burn', 'blur', 'boat', 'bond', 'bulk', 'bust', 'barn', 'bird'],
    'c': ['calm', 'coin', 'cord', 'curl', 'cost', 'crop', 'cult', 'cart', 'clan', 'coal'],
    'd': ['dart', 'disk', 'dock', 'drum', 'dual', 'dusk', 'dawn', 'dirt', 'door', 'dust'],
    'e': ['earn', 'edit', 'elm', 'emit', 'euro', 'exit', 'epic', 'erst', 'each', 'east'],
    'f': ['fact', 'firm', 'fold', 'fork', 'fuel', 'fund', 'fist', 'flat', 'flux', 'form'],
    'g': ['gain', 'glow', 'gold', 'grit', 'gulf', 'gust', 'gift', 'goal', 'grid', 'grip'],
    'h': ['halt', 'hint', 'hold', 'host', 'hull', 'husk', 'halo', 'harm', 'hilt', 'horn'],
    'i': ['icon', 'iron', 'iris', 'idol', 'inch', 'into', 'item', 'iota', 'itch', 'isle'],
    'j': ['join', 'jolt', 'just', 'jury', 'jump', 'jest', 'jolt', 'jamb', 'jolt', 'jolt'],
    'k': ['knot', 'knob', 'kilo', 'kind', 'king', 'kart', 'knit', 'kick', 'kiln', 'kiwi'],
    'l': ['lamp', 'link', 'lock', 'loft', 'lump', 'lush', 'lark', 'limb', 'lion', 'lord'],
    'm': ['mark', 'mint', 'mold', 'musk', 'myth', 'malt', 'mist', 'monk', 'morn', 'must'],
    'n': ['nail', 'nod', 'norm', 'null', 'numb', 'nook', 'nick', 'noon', 'nova', 'nub'],
    'o': ['oath', 'omit', 'oral', 'orb', 'oust', 'oval', 'oxid', 'opal', 'opus', 'orca'],
    'p': ['palm', 'pint', 'plot', 'port', 'pulp', 'push', 'pact', 'pawn', 'plum', 'pond'],
    'q': ['quip', 'quiz', 'quod', 'quilt', 'quart', 'quirk', 'quota', 'quail', 'qualm', 'quash'],
    'r': ['rank', 'rift', 'rock', 'rust', 'ramp', 'rind', 'roam', 'ruin', 'rung', 'rush'],
    's': ['salt', 'silk', 'slot', 'sort', 'spur', 'stir', 'surf', 'slab', 'slim', 'snap'],
    't': ['tact', 'tilt', 'toll', 'tusk', 'turf', 'tarn', 'tick', 'torn', 'trim', 'twig'],
    'u': ['unit', 'upon', 'urns', 'unto', 'ulna', 'umps', 'undo', 'unix', 'ugly', 'ultra'],
    'v': ['vast', 'volt', 'void', 'vow', 'vial', 'vim', 'visa', 'vivid', 'valor', 'vault'],
    'w': ['warp', 'wilt', 'wolf', 'writ', 'wand', 'wink', 'wisp', 'worm', 'writ', 'wrist'],
    'x': ['xray', 'xyst', 'xmas', 'xtra', 'xyst', 'xray', 'xyst', 'xray', 'xyst', 'xray'],
    'y': ['yank', 'yarn', 'yawn', 'yogi', 'yolk', 'your', 'yurt', 'yard', 'yell', 'yoga'],
    'z': ['zap', 'zeal', 'zinc', 'zone', 'zoom', 'zest', 'zero', 'zing', 'zonal', 'zilch'],
}

# Extended word bank with more options per letter (for forbidden letter avoidance)
EXTENDED_WORDS = {
    'a': ['also', 'area', 'art', 'aim', 'arc', 'arm', 'ask', 'atom', 'auto', 'axis',
          'acid', 'acorn', 'adapt', 'admit', 'adopt', 'adult', 'album', 'align', 'altar', 'ample'],
    'b': ['bold', 'born', 'burn', 'blur', 'boat', 'bond', 'bulk', 'bust', 'barn', 'bird',
          'band', 'bark', 'basin', 'batch', 'blank', 'blast', 'bliss', 'block', 'bloom', 'board'],
    'c': ['calm', 'coin', 'cord', 'curl', 'cost', 'crop', 'cult', 'cart', 'clan', 'coal',
          'cabin', 'canal', 'cargo', 'carol', 'chain', 'chalk', 'charm', 'choir', 'civil', 'claim'],
    'd': ['dart', 'disk', 'dock', 'drum', 'dual', 'dusk', 'dawn', 'dirt', 'door', 'dust',
          'dairy', 'damp', 'datum', 'drift', 'drill', 'drink', 'droit', 'druid', 'dwarf', 'droit'],
    'e': ['earn', 'edit', 'elm', 'emit', 'euro', 'exit', 'epic', 'erst', 'each', 'east',
          'elbow', 'elder', 'elect', 'embed', 'enact', 'endow', 'equip', 'erupt', 'evict', 'exalt'],
    'f': ['fact', 'firm', 'fold', 'fork', 'fuel', 'fund', 'fist', 'flat', 'flux', 'form',
          'faith', 'fault', 'feast', 'fiber', 'field', 'final', 'flair', 'flesh', 'float', 'flora'],
    'g': ['gain', 'glow', 'gold', 'grit', 'gulf', 'gust', 'gift', 'goal', 'grid', 'grip',
          'gland', 'glint', 'globe', 'gloom', 'glyph', 'gourd', 'grain', 'grand', 'grasp', 'grind'],
    'h': ['halt', 'hint', 'hold', 'host', 'hull', 'husk', 'halo', 'harm', 'hilt', 'horn',
          'habit', 'harsh', 'haunt', 'haven', 'heart', 'heist', 'hoist', 'honor', 'humid', 'hyper'],
    'i': ['icon', 'iron', 'iris', 'idol', 'inch', 'into', 'item', 'iota', 'itch', 'isle',
          'igloo', 'image', 'imply', 'incur', 'index', 'infer', 'input', 'inter', 'ivory', 'issue'],
    'j': ['join', 'jolt', 'just', 'jury', 'jump', 'jest', 'jamb', 'jaunt', 'jewel', 'joint'],
    'k': ['knot', 'knob', 'kilo', 'kind', 'king', 'kart', 'knit', 'kick', 'kiln', 'kiwi'],
    'l': ['lamp', 'link', 'lock', 'loft', 'lump', 'lush', 'lark', 'limb', 'lion', 'lord',
          'labor', 'lance', 'latch', 'layer', 'ledge', 'lever', 'light', 'linen', 'logic', 'lunar'],
    'm': ['mark', 'mint', 'mold', 'musk', 'myth', 'malt', 'mist', 'monk', 'morn', 'must',
          'magic', 'manor', 'marsh', 'medal', 'mercy', 'micro', 'might', 'minor', 'modal', 'moral'],
    'n': ['nail', 'nod', 'norm', 'null', 'numb', 'nook', 'nick', 'noon', 'nova', 'nub',
          'nadir', 'naval', 'nerve', 'night', 'noble', 'north', 'notch', 'novel', 'nudge', 'nurse'],
    'o': ['oath', 'omit', 'oral', 'orb', 'oust', 'oval', 'oxid', 'opal', 'opus', 'orca',
          'ocean', 'olive', 'onset', 'opera', 'orbit', 'organ', 'other', 'outer', 'outdo', 'oxide'],
    'p': ['palm', 'pint', 'plot', 'port', 'pulp', 'push', 'pact', 'pawn', 'plum', 'pond',
          'panel', 'patch', 'pearl', 'pedal', 'pilot', 'pixel', 'plank', 'plaza', 'plumb', 'polar'],
    'q': ['quip', 'quiz', 'quod', 'quilt', 'quart', 'quirk', 'quota', 'quail', 'qualm', 'quash'],
    'r': ['rank', 'rift', 'rock', 'rust', 'ramp', 'rind', 'roam', 'ruin', 'rung', 'rush',
          'radar', 'rally', 'ranch', 'rapid', 'realm', 'reign', 'relay', 'rigid', 'rival', 'robin'],
    's': ['salt', 'silk', 'slot', 'sort', 'spur', 'stir', 'surf', 'slab', 'slim', 'snap',
          'saint', 'salon', 'satin', 'scout', 'shaft', 'shelf', 'shift', 'shrub', 'sigma', 'skull'],
    't': ['tact', 'tilt', 'toll', 'tusk', 'turf', 'tarn', 'tick', 'torn', 'trim', 'twig',
          'talon', 'tempo', 'thorn', 'tiger', 'titan', 'token', 'torch', 'tower', 'trail', 'trout'],
    'u': ['unit', 'upon', 'urns', 'unto', 'ulna', 'umps', 'undo', 'unix', 'ugly', 'ultra',
          'umbra', 'uncut', 'under', 'union', 'unity', 'until', 'upper', 'urban', 'usher', 'utter'],
    'v': ['vast', 'volt', 'void', 'vow', 'vial', 'vim', 'visa', 'vivid', 'valor', 'vault',
          'vapor', 'vigor', 'vinyl', 'viral', 'visor', 'vital', 'vivid', 'vocal', 'voila', 'voter'],
    'w': ['warp', 'wilt', 'wolf', 'writ', 'wand', 'wink', 'wisp', 'worm', 'writ', 'wrist',
          'waist', 'watch', 'wheat', 'whirl', 'width', 'witch', 'world', 'worth', 'wound', 'wraith'],
    'x': ['xray', 'xyst'],
    'y': ['yank', 'yarn', 'yawn', 'yogi', 'yolk', 'your', 'yurt', 'yard', 'yell', 'yoga'],
    'z': ['zap', 'zeal', 'zinc', 'zone', 'zoom', 'zest', 'zero', 'zing', 'zonal', 'zilch'],
}


class ArtifactGenerator:
    """Generate a single-line artifact string satisfying all constraints.

    Strategy:
    1. Compute all required elements
    2. Place elements matching acrostic letters in acrostic positions
    3. Fill remaining acrostic positions with filler words
    4. Place remaining required elements after acrostic
    5. Fill to target word count
    6. Verify forbidden letter constraint
    """

    def __init__(self, constraint_info: dict):
        self.word_count = constraint_info.get('word_count', 16)
        self.required_elements = constraint_info.get('required_elements', [])
        self.acrostic = constraint_info.get('acrostic', '')
        self.forbidden_letter = constraint_info.get('forbidden_letter', '')
        self.equation = constraint_info.get('equation', '')

    def generate(self) -> str:
        """Generate the artifact string."""
        dbg(f"\n=== GENERATING ARTIFACT ===")
        dbg(f"  Target words: {self.word_count}")
        dbg(f"  Acrostic: '{self.acrostic}'")
        dbg(f"  Forbidden: '{self.forbidden_letter}'")
        dbg(f"  Required: {self.required_elements}")

        acrostic_len = len(self.acrostic)
        words = [''] * self.word_count

        # Track which required elements have been placed
        placed = set()

        # Step 1: Place required elements in acrostic positions if they match
        if acrostic_len > 0:
            for i in range(min(acrostic_len, self.word_count)):
                target_letter = self.acrostic[i].lower()
                # Check if any required element starts with this letter
                for elem in self.required_elements:
                    if elem.lower().startswith(target_letter) and elem not in placed:
                        # Check if element is a single word
                        if ' ' not in elem:
                            words[i] = elem
                            placed.add(elem)
                            dbg(f"  Placed '{elem}' at acrostic position {i} (letter '{target_letter}')")
                            break

        # Step 2: Fill remaining acrostic positions with filler words
        if acrostic_len > 0:
            for i in range(min(acrostic_len, self.word_count)):
                if words[i]:
                    continue
                target_letter = self.acrostic[i].lower()
                filler = self._get_filler_word(target_letter, words)
                words[i] = filler
                dbg(f"  Filled acrostic position {i} with '{filler}' (letter '{target_letter}')")

        # Step 3: Place remaining required elements (multi-word elements need special handling)
        remaining = [e for e in self.required_elements if e not in placed]
        pos = acrostic_len  # Start placing after acrostic positions

        for elem in remaining:
            elem_words = elem.split()
            if pos + len(elem_words) <= self.word_count:
                for j, ew in enumerate(elem_words):
                    words[pos + j] = ew
                placed.add(elem)
                dbg(f"  Placed '{elem}' at positions {pos}-{pos+len(elem_words)-1}")
                pos += len(elem_words)
            elif pos < self.word_count:
                # Try to fit at least part of it
                words[pos] = elem.replace(' ', '')  # Concatenate as single word
                placed.add(elem)
                pos += 1

        # Step 4: Fill remaining positions with filler words
        for i in range(self.word_count):
            if not words[i]:
                filler = self._get_filler_word(None, words)
                words[i] = filler

        # Step 5: Verify and fix forbidden letter
        if self.forbidden_letter:
            words = self._fix_forbidden_letter(words)

        artifact = ' '.join(words)
        dbg(f"\n  ARTIFACT: {artifact}")
        dbg(f"  Word count: {len(artifact.split())}")

        return artifact

    def _get_filler_word(self, start_letter: Optional[str], existing_words: List[str]) -> str:
        """Get a filler word, optionally starting with a specific letter.

        Avoids the forbidden letter and tries not to repeat words.
        """
        forbidden = self.forbidden_letter.lower() if self.forbidden_letter else ''
        used = set(w.lower() for w in existing_words if w)

        if start_letter:
            sl = start_letter.lower()
            # Try extended word bank first
            candidates = EXTENDED_WORDS.get(sl, WORD_BANK.get(sl, []))
            for word in candidates:
                if forbidden and forbidden in word.lower():
                    continue
                if word.lower() not in used:
                    return word
            # If all used, allow repeats but still avoid forbidden
            for word in candidates:
                if forbidden and forbidden in word.lower():
                    continue
                return word
            # Last resort: construct a word
            return self._construct_word(sl, forbidden)
        else:
            # No specific start letter needed — pick any safe filler
            safe_fillers = [
                'also', 'bold', 'calm', 'dart', 'earn', 'firm', 'gain', 'halt',
                'icon', 'join', 'knot', 'lamp', 'mark', 'nail', 'oath', 'palm',
                'rank', 'salt', 'tact', 'unit', 'vast', 'warp', 'yarn', 'zeal',
                'form', 'gold', 'hint', 'iron', 'link', 'mist', 'norm', 'plot',
                'rock', 'silk', 'tilt', 'void', 'wilt', 'zinc', 'atom', 'bond',
            ]
            for word in safe_fillers:
                if forbidden and forbidden in word.lower():
                    continue
                if word.lower() not in used:
                    return word
            # Allow repeats
            for word in safe_fillers:
                if forbidden and forbidden in word.lower():
                    continue
                return word
            return 'ok'

    def _construct_word(self, start: str, forbidden: str) -> str:
        """Construct a simple word starting with the given letter, avoiding forbidden letter."""
        vowels = [v for v in 'aeiou' if v != forbidden]
        consonants = [c for c in 'bcdfghlmnprst' if c != forbidden]

        if not vowels:
            vowels = ['a']
        if not consonants:
            consonants = ['n']

        # If start letter IS the forbidden letter, pick a safe alternative
        if start.lower() == forbidden.lower():
            start = consonants[0] if start not in 'aeiou' else vowels[0]

        if start in 'aeiou':
            return start + consonants[0] + vowels[0] + consonants[1 % len(consonants)]
        else:
            return start + vowels[0] + consonants[0] + vowels[1 % len(vowels)]

    def _fix_forbidden_letter(self, words: List[str]) -> List[str]:
        """Replace words containing the forbidden letter with safe alternatives."""
        forbidden = self.forbidden_letter.lower()
        result = []

        for i, word in enumerate(words):
            if forbidden in word.lower():
                # This word contains the forbidden letter — need replacement
                start = word[0].lower() if word else 'a'

                # Check if this is a required element (can't easily replace)
                is_required = any(word.lower() in elem.lower() for elem in self.required_elements)

                if is_required:
                    # Required element contains forbidden letter — try to find synonym
                    # For now, keep it (the challenge may expect this)
                    dbg(f"  WARNING: Required element '{word}' contains forbidden letter '{forbidden}'")
                    result.append(word)
                else:
                    # Replace with safe filler
                    if i < len(self.acrostic):
                        # Must maintain acrostic letter
                        replacement = self._get_filler_word(self.acrostic[i].lower(), words)
                    else:
                        replacement = self._get_filler_word(None, words + result)
                    # Verify replacement is safe
                    if forbidden in replacement.lower():
                        replacement = self._construct_word(start, forbidden)
                    result.append(replacement)
                    dbg(f"  Replaced '{word}' with '{replacement}' (forbidden '{forbidden}')")
            else:
                result.append(word)

        return result

    def validate(self, artifact: str) -> Tuple[bool, List[str]]:
        """Validate the artifact against all constraints.

        Returns (all_passed, list_of_failure_messages).
        """
        errors = []
        words = artifact.split()

        # Word count
        if len(words) != self.word_count:
            errors.append(f"Word count: expected {self.word_count}, got {len(words)}")

        # Acrostic
        if self.acrostic:
            actual = ''.join(w[0].upper() for w in words[:len(self.acrostic)])
            if actual != self.acrostic.upper():
                errors.append(f"Acrostic: expected '{self.acrostic.upper()}', got '{actual}'")

        # Required elements
        artifact_lower = artifact.lower()
        for elem in self.required_elements:
            if elem.lower() not in artifact_lower:
                errors.append(f"Missing required element: '{elem}'")

        # Forbidden letter
        if self.forbidden_letter:
            fl = self.forbidden_letter.lower()
            # Check non-required words
            for word in words:
                if fl in word.lower():
                    is_req = any(word.lower() in elem.lower() for elem in self.required_elements)
                    if not is_req:
                        errors.append(f"Forbidden letter '{fl}' found in non-required word '{word}'")

        all_passed = len(errors) == 0
        return all_passed, errors


# =============================================================================
# MAIN SOLVER
# =============================================================================

def extract_company_names_from_doc(doc: str) -> List[str]:
    """Extract company names from ENTITY and FILING lines in the document."""
    names = []
    for line in doc.split('\n'):
        line = line.strip()
        # ENTITY: Full Name / Short | ...
        m = re.match(r'ENTITY:\s*(.+?)\s*/\s*\w+\s*\|', line)
        if m:
            names.append(m.group(1).strip())
            continue
        # FILING: Full Name (DBA: XX) | ...
        m = re.match(r'FILING:\s*(.+?)\s*\(DBA:', line)
        if m:
            names.append(m.group(1).strip())
            continue
    return names


def solve(challenge: dict) -> str:
    """Solve a BOTCOIN mining challenge.

    Args:
        challenge: Dict with keys 'doc', 'questions', 'constraints'

    Returns:
        The artifact string (single line).
    """
    doc = challenge.get('doc', '')
    questions = challenge.get('questions', [])
    constraints = challenge.get('constraints', [])

    dbg("=" * 60)
    dbg("BOTCOIN SOLVER v2 — Deterministic Engine")
    dbg("=" * 60)
    dbg(f"Document length: {len(doc)} chars")
    dbg(f"Questions: {len(questions)}")
    dbg(f"Constraints: {len(constraints)}")

    # Step 1: Extract company names from document
    company_names = extract_company_names_from_doc(doc)
    # Fallback: use challenge['companies'] if doc-based extraction finds nothing
    if not company_names:
        company_names = challenge.get('companies', [])
        dbg(f"\nUsing challenge companies field: {len(company_names)} names")
    dbg(f"\nExtracted {len(company_names)} company names: {company_names}")

    # Step 2: Parse document
    parser = DocumentParser(doc, company_names)
    companies = parser.parse()

    if not companies:
        dbg("WARNING: No companies parsed!")
        return "no data found in document"

    # Step 3: Answer questions
    engine = QuestionEngine(companies)
    answers = []
    for i, q in enumerate(questions):
        ans = engine.answer(q)
        answers.append(ans)
        dbg(f"\n  Q{i+1}: {q}")
        dbg(f"  A{i+1}: {ans}")

    # Step 4: Compute constraints
    constraint_engine = ConstraintEngine(constraints, questions, answers, companies)
    constraint_info = constraint_engine.compute_all()

    # Step 5: Generate artifact
    generator = ArtifactGenerator(constraint_info)
    artifact = generator.generate()

    # Step 6: Validate
    passed, errors = generator.validate(artifact)
    dbg(f"\n=== VALIDATION ===")
    if passed:
        dbg("  ALL CHECKS PASSED")
    else:
        for err in errors:
            dbg(f"  FAIL: {err}")

        # Try to fix issues
        if errors:
            dbg("\n  Attempting fixes...")
            artifact = _attempt_fix(artifact, constraint_info, errors)
            passed2, errors2 = generator.validate(artifact)
            if passed2:
                dbg("  FIXES SUCCESSFUL")
            else:
                for err in errors2:
                    dbg(f"  STILL FAILING: {err}")

    return artifact


def _attempt_fix(artifact: str, constraint_info: dict, errors: List[str]) -> str:
    """Attempt to fix validation errors in the artifact."""
    words = artifact.split()
    target_count = constraint_info.get('word_count', len(words))
    forbidden = constraint_info.get('forbidden_letter', '')
    acrostic = constraint_info.get('acrostic', '')
    required = constraint_info.get('required_elements', [])

    # Fix word count
    while len(words) < target_count:
        filler = 'also'
        if forbidden and forbidden in filler:
            filler = 'bold'
        if forbidden and forbidden in filler:
            filler = 'calm'
        if forbidden and forbidden in filler:
            filler = 'dart'
        words.append(filler)

    while len(words) > target_count:
        # Remove from the end, but not required elements
        if len(words) > len(acrostic):
            last = words[-1]
            is_req = any(last.lower() in elem.lower() for elem in required)
            if not is_req:
                words.pop()
            else:
                # Remove second-to-last non-required word
                for i in range(len(words) - 2, len(acrostic) - 1, -1):
                    is_req_i = any(words[i].lower() in elem.lower() for elem in required)
                    if not is_req_i:
                        words.pop(i)
                        break
                else:
                    words.pop()  # Last resort
        else:
            break

    # Fix missing required elements
    artifact_lower = ' '.join(words).lower()
    for elem in required:
        if elem.lower() not in artifact_lower:
            # Try to insert the element
            elem_words = elem.split()
            # Find non-acrostic, non-required positions to replace
            for i in range(len(acrostic), len(words)):
                is_req_i = any(words[i].lower() in e.lower() for e in required if e != elem)
                if not is_req_i and i + len(elem_words) - 1 < len(words):
                    for j, ew in enumerate(elem_words):
                        words[i + j] = ew
                    break

    return ' '.join(words)


def main():
    """Entry point: read challenge from stdin, output artifact to stdout."""
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        run_tests()
        return

    # Read challenge JSON from stdin
    try:
        raw = sys.stdin.read()
        challenge = json.loads(raw)
    except json.JSONDecodeError as e:
        dbg(f"ERROR: Invalid JSON input: {e}")
        sys.exit(1)

    artifact = solve(challenge)
    # Output ONLY the artifact to stdout
    print(artifact)


# =============================================================================
# SELF-TESTS
# =============================================================================

def run_tests():
    """Run self-tests for all parsing and computation functions."""
    passed = 0
    failed = 0

    def check(name, actual, expected):
        nonlocal passed, failed
        if actual == expected:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name}: expected {expected!r}, got {actual!r}")
            failed += 1

    print("=== REVENUE PARSING TESTS ===")
    check("$4,356 million", parse_revenue_amount("$4,356 million"), 4356)
    check("$4,356M", parse_revenue_amount("$4,356M"), 4356)
    check("4,356M", parse_revenue_amount("4,356M"), 4356)
    check("$4.88 billion", parse_revenue_amount("$4.88 billion"), 4880)
    check("$4.88B", parse_revenue_amount("$4.88B"), 4880)
    check("$1.58 billion", parse_revenue_amount("$1.58 billion"), 1580)
    check("close to 2475 million dollars", parse_revenue_amount("close to 2475 million dollars"), 2475)
    check("approximately 1312 million dollars", parse_revenue_amount("approximately 1312 million dollars"), 1312)
    check("just under 3651 million dollars", parse_revenue_amount("just under 3651 million dollars"), 3651)
    check("just over 1642 million dollars", parse_revenue_amount("just over 1642 million dollars"), 1642)
    check("nearly 462 million dollars", parse_revenue_amount("nearly 462 million dollars"), 462)
    check("roughly 3268 million dollars", parse_revenue_amount("roughly 3268 million dollars"), 3268)
    check("about 447 million dollars", parse_revenue_amount("about 447 million dollars"), 447)
    check("$1,898 million", parse_revenue_amount("$1,898 million"), 1898)
    check("138M", parse_revenue_amount("138M"), 138)
    check("$3,607M", parse_revenue_amount("$3,607M"), 3607)
    check("$2.5 billion", parse_revenue_amount("$2.5 billion"), 2500)
    check("$0.5 billion", parse_revenue_amount("$0.5 billion"), 500)

    print("\n=== EMPLOYEE COUNT PARSING TESTS ===")
    check("41,819", parse_employee_count("41,819"), 41819)
    check("approximately 41819", parse_employee_count("approximately 41819"), 41819)
    check("nearly 14092", parse_employee_count("nearly 14092"), 14092)
    check("just over 5000", parse_employee_count("just over 5000"), 5000)
    check("close to 8633", parse_employee_count("close to 8633"), 8633)

    print("\n=== GROWTH RATE PARSING TESTS ===")
    check("+25%", parse_growth_rate("+25%"), 25.0)
    check("-12%", parse_growth_rate("-12%"), -12.0)
    check("0%", parse_growth_rate("0%"), 0.0)
    check("+4%", parse_growth_rate("+4%"), 4.0)
    check("(+35%)", parse_growth_rate("(+35%)"), 35.0)
    check("(-20%)", parse_growth_rate("(-20%)"), -20.0)

    print("\n=== PRIME NUMBER TESTS ===")
    check("next_prime(11)", next_prime(11), 11)
    check("next_prime(12)", next_prime(12), 13)
    check("next_prime(14)", next_prime(14), 17)
    check("next_prime(20)", next_prime(20), 23)
    check("next_prime(100)", next_prime(100), 101)
    check("next_prime(110)", next_prime(110), 113)
    # Constraint: (employees % 100) + 11
    check("emp=41819 → prime", next_prime((41819 % 100) + 11), 31)  # 19+11=30, next prime=31
    check("emp=14092 → prime", next_prime((14092 % 100) + 11), 103)  # 92+11=103, is prime

    print("\n=== EQUATION TESTS ===")
    # A = (Q1_rev % 90) + 10, B = (Q4_rev % 90) + 10
    q1, q4 = 4356, 3607
    a = (q1 % 90) + 10
    b = (q4 % 90) + 10
    c = a + b
    check(f"equation A (Q1={q1})", a, 46)  # 4356%90=36, +10=46
    check(f"equation B (Q4={q4})", b, 17)  # 3607%90=7, +10=17
    check(f"equation C", c, 63)
    check(f"equation string", f"{a}+{b}={c}", "46+17=63")

    print("\n=== INITIALS TESTS ===")
    check("Xeno Logic", get_initials("Xeno Logic"), "XL")
    check("Torq Hub", get_initials("Torq Hub"), "TH")
    check("Coda Data", get_initials("Coda Data"), "CD")
    check("Apex Nexus", get_initials("Apex Nexus"), "AN")

    print("\n=== COUNTERFACTUAL DETECTION TESTS ===")
    check("counterfactual line", is_counterfactual(
        "If Dyna Net had pursued a rumored merger, analysts once claimed the closing quarter could have looked like $2,802 million — a counterfactual that never materialized."
    ), True)
    check("normal line", is_counterfactual(
        "Pyra Dynamics REVENUE: first quarter $4,356 million; Q2 $1,898 million"
    ), False)
    check("chatter line", is_counterfactual(
        "Early chatter suggested Torq might 'double headcount overnight' after a supposed acquisition; that claim was later walked back."
    ), True)

    print("\n=== QUARTER IDENTIFICATION TESTS ===")
    check("Q1", identify_quarter("Q1"), 0)
    check("Q2", identify_quarter("Q2"), 1)
    check("Q3", identify_quarter("Q3"), 2)
    check("Q4", identify_quarter("Q4"), 3)
    check("first quarter", identify_quarter("first quarter"), 0)
    check("second quarter", identify_quarter("second quarter"), 1)
    check("third quarter", identify_quarter("third quarter"), 2)
    check("fourth quarter", identify_quarter("fourth quarter"), 3)
    check("opening quarter", identify_quarter("opening quarter"), 0)
    check("closing quarter", identify_quarter("closing quarter"), 3)
    check("mid-spring quarter", identify_quarter("mid-spring quarter"), 1)
    check("follow-on quarter", identify_quarter("follow-on quarter"), 1)
    check("pre-close quarter", identify_quarter("pre-close quarter"), 2)
    check("late-year quarter", identify_quarter("late-year quarter"), 2)
    check("year-end quarter", identify_quarter("year-end quarter"), 3)

    print(f"\n=== RESULTS: {passed} passed, {failed} failed ===")
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
