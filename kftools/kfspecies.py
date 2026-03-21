import re
from dataclasses import dataclass


SUPPORTED_SPECIES_PARSERS = ("legacy", "taxonomic")
_CANDIDATE_SPLIT_RE = re.compile(r"[|@:;,\s=]+")
_PROXIMITY_QUALIFIERS = frozenset(("cf", "aff", "nr"))
_GENUS_ONLY_PLACEHOLDERS = frozenset(("sp", "sp.", "spp", "spp."))
_RANK_ALIASES = {
    "subsp": "subsp",
    "ssp": "subsp",
    "subspecies": "subsp",
    "var": "var",
    "variety": "var",
    "forma": "forma",
    "form": "forma",
    "f": "forma",
    "strain": "strain",
    "substrain": "substrain",
    "serovar": "serovar",
    "serotype": "serotype",
    "serogroup": "serogroup",
    "pathovar": "pathovar",
    "pv": "pathovar",
    "biovar": "biovar",
    "biotype": "biotype",
    "chemovar": "chemovar",
    "morphovar": "morphovar",
    "cultivar": "cultivar",
    "cv": "cultivar",
    "isolate": "isolate",
    "group": "group",
    "subgroup": "subgroup",
    "complex": "complex",
    "clade": "clade",
    "lineage": "lineage",
    "section": "section",
    "series": "series",
    "ecotype": "ecotype",
    "breed": "breed",
}
_DISPLAY_RANKS = {
    "subsp": "subsp.",
    "var": "var.",
    "forma": "f.",
}


@dataclass(frozen=True)
class SpeciesParseResult:
    species_label: str
    scientific_name: str = None
    taxonomy_query: str = None

    def __post_init__(self):
        normalized_species_label = _normalize_species_label(self.species_label)
        if normalized_species_label in (None, ""):
            raise ValueError("species_label must be a non-empty string")
        object.__setattr__(self, "species_label", normalized_species_label)
        if self.scientific_name is None:
            object.__setattr__(
                self,
                "scientific_name",
                _scientific_name_from_species_label(normalized_species_label),
            )
        if self.taxonomy_query is None:
            object.__setattr__(
                self,
                "taxonomy_query",
                _taxonomy_query_from_species_label(normalized_species_label),
            )

    @property
    def genus(self):
        return self.species_label.split("_")[0]

    @property
    def species(self):
        parts = self.species_label.split("_")
        if len(parts) < 2:
            return ""
        if parts[1].lower() == "sp":
            return parts[2] if len(parts) >= 3 else "sp"
        if parts[1].lower() in _PROXIMITY_QUALIFIERS:
            return parts[2] if len(parts) >= 3 else parts[1]
        return parts[1]


def _validate_species_label_input(label):
    if (not isinstance(label, str)) or (label.strip() == ""):
        raise ValueError("label must be a non-empty string")


def _normalize_species_label(text):
    if text is None:
        return None
    normalized = re.sub(r"\s+", "_", str(text).strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or None


def _normalize_genus_token(token):
    cleaned = str(token or "").strip()
    if cleaned == "":
        raise ValueError("genus token must not be empty")
    return cleaned[:1].upper() + cleaned[1:].lower()


def _normalize_rank_value(token):
    cleaned = str(token or "").strip()
    if cleaned == "":
        raise ValueError("rank value token must not be empty")
    return cleaned


def _canonical_taxonomic_token(token):
    cleaned = str(token or "").strip()
    lowered = cleaned.lower()
    if lowered in _GENUS_ONLY_PLACEHOLDERS:
        return "sp"
    if lowered in _PROXIMITY_QUALIFIERS:
        return lowered
    if lowered in _RANK_ALIASES:
        return _RANK_ALIASES[lowered]
    return cleaned


def _scientific_name_from_species_label(species_label):
    parts = [part for part in str(species_label).split("_") if part != ""]
    if len(parts) >= 3 and parts[1].lower() in _PROXIMITY_QUALIFIERS:
        return "{} {}. {}".format(parts[0], parts[1].lower(), parts[2])
    if len(parts) >= 3 and parts[1].lower() == "sp":
        return "{} sp. {}".format(parts[0], parts[2])
    if len(parts) >= 4 and parts[2].lower() in _RANK_ALIASES.values():
        rank = parts[2].lower()
        return "{} {} {} {}".format(
            parts[0],
            parts[1],
            _DISPLAY_RANKS.get(rank, rank),
            parts[3],
        )
    return species_label.replace("_", " ")


def _taxonomy_query_from_species_label(species_label):
    parts = [part for part in str(species_label).split("_") if part != ""]
    if len(parts) >= 3 and parts[1].lower() == "sp":
        return parts[0]
    if len(parts) >= 3 and parts[1].lower() in _PROXIMITY_QUALIFIERS:
        return "{} {}".format(parts[0], parts[2])
    if len(parts) >= 4 and parts[2].lower() in _RANK_ALIASES.values():
        return "{} {}".format(parts[0], parts[1])
    if len(parts) >= 2:
        return "{} {}".format(parts[0], parts[1])
    return species_label.replace("_", " ")


def _parse_legacy_text(text):
    normalized_label = _normalize_species_label(text)
    if normalized_label is None:
        raise ValueError("parsed species text must not be empty")
    parts = [part for part in normalized_label.split("_") if part != ""]
    if len(parts) < 2:
        raise ValueError("label must contain at least two underscore-delimited tokens")
    if (parts[0] == "") or (parts[1] == ""):
        raise ValueError("label must contain non-empty genus and species tokens")
    return SpeciesParseResult(
        species_label="{}_{}".format(parts[0], parts[1]),
    )


def _parse_taxonomic_tokens(tokens):
    if len(tokens) < 2:
        raise ValueError("label must contain at least two underscore-delimited tokens")
    normalized_tokens = [_canonical_taxonomic_token(token) for token in tokens if str(token).strip() != ""]
    if len(normalized_tokens) < 2:
        raise ValueError("label must contain at least two underscore-delimited tokens")
    genus = _normalize_genus_token(normalized_tokens[0])
    second = normalized_tokens[1]
    second_lower = second.lower()

    if second_lower in _PROXIMITY_QUALIFIERS:
        if len(normalized_tokens) < 3:
            raise ValueError("taxonomic proximity qualifiers require a species token")
        species = str(normalized_tokens[2]).lower()
        return (
            SpeciesParseResult(
                species_label="{}_{}_{}".format(genus, second_lower, species),
                scientific_name="{} {}. {}".format(genus, second_lower, species),
                taxonomy_query="{} {}".format(genus, species),
            ),
            3,
        )

    if second_lower == "sp":
        if len(normalized_tokens) >= 3:
            label_token = _normalize_rank_value(normalized_tokens[2])
            return (
                SpeciesParseResult(
                    species_label="{}_sp_{}".format(genus, label_token),
                    scientific_name="{} sp. {}".format(genus, label_token),
                    taxonomy_query=genus,
                ),
                3,
            )
        return (
            SpeciesParseResult(
                species_label="{}_sp".format(genus),
                scientific_name="{} sp.".format(genus),
                taxonomy_query=genus,
            ),
            2,
        )

    species = str(second).lower()
    if len(normalized_tokens) >= 3:
        third = normalized_tokens[2]
        third_lower = str(third).lower()
        if third_lower in _PROXIMITY_QUALIFIERS:
            return (
                SpeciesParseResult(
                    species_label="{}_{}_{}".format(genus, third_lower, species),
                    scientific_name="{} {}. {}".format(genus, third_lower, species),
                    taxonomy_query="{} {}".format(genus, species),
                ),
                3,
            )
        if third_lower in _RANK_ALIASES.values():
            if len(normalized_tokens) < 4:
                raise ValueError("infraspecific ranks require a trailing label token")
            rank = third_lower
            value = _normalize_rank_value(normalized_tokens[3])
            return (
                SpeciesParseResult(
                    species_label="{}_{}_{}_{}".format(genus, species, rank, value),
                    scientific_name="{} {} {} {}".format(
                        genus,
                        species,
                        _DISPLAY_RANKS.get(rank, rank),
                        value,
                    ),
                    taxonomy_query="{} {}".format(genus, species),
                ),
                4,
            )

    return (
        SpeciesParseResult(
            species_label="{}_{}".format(genus, species),
            scientific_name="{} {}".format(genus, species),
            taxonomy_query="{} {}".format(genus, species),
        ),
        2,
    )


def _parse_taxonomic_text(text):
    normalized_label = _normalize_species_label(text)
    if normalized_label is None:
        raise ValueError("parsed species text must not be empty")
    candidate_rows = []
    for fragment_index, fragment in enumerate(_CANDIDATE_SPLIT_RE.split(normalized_label)):
        if fragment == "":
            continue
        fragment_tokens = [part for part in fragment.split("_") if part != ""]
        if len(fragment_tokens) < 2:
            continue
        try:
            parsed_result, consumed_tokens = _parse_taxonomic_tokens(fragment_tokens)
        except ValueError:
            continue
        starts_with_uppercase = fragment_tokens[0][0].isalpha() and fragment_tokens[0][0].isupper()
        candidate_rows.append(
            (
                consumed_tokens,
                starts_with_uppercase,
                -fragment_index,
                parsed_result,
            )
        )
    if len(candidate_rows) == 0:
        raise ValueError("label did not contain a taxonomic species token")
    return max(candidate_rows, key=lambda row: (row[0], row[1], row[2]))[3]


def _parse_species_text(text):
    if not isinstance(text, str):
        raise ValueError("parsed species text must be a string")
    text = text.strip()
    if text == "":
        raise ValueError("parsed species text must not be empty")
    try:
        return _parse_taxonomic_text(text)
    except ValueError:
        return _parse_legacy_text(text)


def _coerce_parse_result(result):
    if isinstance(result, SpeciesParseResult):
        return result
    if isinstance(result, dict):
        if ("genus" in result) and ("species" in result):
            return _parse_species_text("{}_{}".format(result["genus"], result["species"]))
        for key in ("species_label", "label", "scientific_name", "name"):
            if key in result:
                return _parse_species_text(str(result[key]))
        raise ValueError(
            "parser result dict must contain genus/species or a species label"
        )
    if isinstance(result, (tuple, list)):
        if len(result) < 2:
            raise ValueError("parser result sequence must contain genus and species")
        return _parse_species_text("{}_{}".format(result[0], result[1]))
    if isinstance(result, str):
        return _parse_species_text(result)
    raise ValueError(
        "parser result must be a string, sequence, dict, or SpeciesParseResult"
    )


def _parse_legacy(label):
    return _parse_legacy_text(label)


def _parse_taxonomic(label):
    return _parse_taxonomic_text(label)


def _build_regex_parser(pattern, group=None):
    compiled_pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
    if not hasattr(compiled_pattern, "search"):
        raise ValueError("regex species parser pattern must be a string or compiled regex")

    def _parse_regex(label):
        match = compiled_pattern.search(label)
        if match is None:
            raise ValueError("label did not match regex species parser")
        if group is not None:
            if isinstance(group, (tuple, list)):
                if len(group) != 2:
                    raise ValueError("regex species parser group sequence must have length 2")
                return _parse_species_text(
                    "{}_{}".format(match.group(group[0]), match.group(group[1]))
                )
            return _parse_species_text(str(match.group(group)))
        group_dict = match.groupdict()
        if ("genus" in group_dict) and ("species" in group_dict):
            return _parse_species_text(
                "{}_{}".format(group_dict["genus"], group_dict["species"])
            )
        if match.lastindex is None:
            return _parse_species_text(match.group(0))
        if match.lastindex >= 2:
            return _parse_species_text(
                "{}_{}".format(match.group(1), match.group(2))
            )
        return _parse_species_text(str(match.group(1)))

    return _parse_regex


def _build_map_parser(mapping):
    if not isinstance(mapping, dict):
        raise ValueError("map species parser requires a dict mapping")

    def _parse_map(label):
        if label not in mapping:
            raise ValueError("label did not match map species parser")
        return _coerce_parse_result(mapping[label])

    return _parse_map


def _coerce_species_parser(species_parser):
    if species_parser is None:
        return _parse_legacy
    if callable(species_parser):
        return species_parser
    if hasattr(species_parser, "search"):
        return _build_regex_parser(species_parser)
    if isinstance(species_parser, str):
        parser_mode = species_parser.strip()
        if parser_mode == "legacy":
            return _parse_legacy
        if parser_mode == "taxonomic":
            return _parse_taxonomic
        return _build_regex_parser(parser_mode)
    if isinstance(species_parser, dict):
        parser_mode = species_parser.get("type") or species_parser.get("mode") or species_parser.get("name")
        if parser_mode in (None, "regex"):
            if "pattern" not in species_parser:
                raise ValueError("regex species parser dict must include a pattern")
            return _build_regex_parser(
                species_parser["pattern"],
                group=species_parser.get("group") or species_parser.get("groups"),
            )
        if parser_mode == "legacy":
            return _parse_legacy
        if parser_mode == "taxonomic":
            return _parse_taxonomic
        if parser_mode == "map":
            return _build_map_parser(species_parser.get("mapping"))
        raise ValueError(
            "Unknown species parser mode: {}. Expected one of: {}".format(
                parser_mode, ", ".join(SUPPORTED_SPECIES_PARSERS + ("regex", "map"))
            )
        )
    if isinstance(species_parser, (tuple, list)):
        if len(species_parser) == 0:
            raise ValueError("species parser sequence must not be empty")
        parser_mode = species_parser[0]
        if parser_mode == "regex":
            if len(species_parser) < 2:
                raise ValueError("regex species parser sequence must include a pattern")
            group = species_parser[2] if len(species_parser) >= 3 else None
            return _build_regex_parser(species_parser[1], group=group)
        if parser_mode == "map":
            if len(species_parser) < 2:
                raise ValueError("map species parser sequence must include a mapping")
            return _build_map_parser(species_parser[1])
        if (len(species_parser) == 1) and (parser_mode in SUPPORTED_SPECIES_PARSERS):
            return _coerce_species_parser(parser_mode)
    raise ValueError(
        "species_parser must be None, a parser name, a regex, a parser config, or a callable"
    )


def parse_species_label(label, species_parser=None, parser=None):
    _validate_species_label_input(label)
    if parser is not None:
        if species_parser is not None:
            raise ValueError("Use only one of species_parser or parser")
        species_parser = parser
    parser = _coerce_species_parser(species_parser)
    return _coerce_parse_result(parser(label))
