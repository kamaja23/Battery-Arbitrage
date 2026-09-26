from __future__ import annotations
import pytest

from wattson.zones import (
    HUBS,
    LOAD_ZONES,
    SETTLEMENT_POINTS,
    label_for,
)


class TestCoverage:
    def test_it_covers_every_zone_and_hub_the_api_serves(self):
        assert set(LOAD_ZONES) == {
            "LZ_AEN", "LZ_CPS", "LZ_HOUSTON", "LZ_LCRA",
            "LZ_NORTH", "LZ_RAYBN", "LZ_SOUTH", "LZ_WEST",
        }
        assert set(HUBS) == {
            "HB_HOUSTON", "HB_NORTH", "HB_PAN", "HB_SOUTH", "HB_WEST",
        }
        assert len(LOAD_ZONES) + len(HUBS) == len(SETTLEMENT_POINTS)


class TestLabels:
    def test_labels_name_a_place_rather_than_just_the_code(self):
        for code in LOAD_ZONES:
            label = label_for(code)
            assert code in label, f"{code} lost its raw code"
            # A readable name and a location, not just the code.
            name, _, area = label.partition(" (")
            assert name and area.strip(") ·"), f"{code} has no place name"

    def test_the_municipal_utility_zones_name_their_utility_and_city(self):
        assert label_for("LZ_AEN").startswith("Austin Energy")
        assert "Austin" in label_for("LZ_AEN")
        assert label_for("LZ_CPS").startswith("CPS Energy")
        assert "San Antonio" in label_for("LZ_CPS")

    def test_the_rayburn_zone_is_labelled_as_a_cooperative_not_a_city(self):
        label = label_for("LZ_RAYBN")
        assert "Rayburn" in label
        assert "Co-op" in label
        assert "East Texas" in label

    def test_competitive_zones_read_as_regions(self):
        assert label_for("LZ_NORTH").startswith("North Texas")
        assert "Dallas" in label_for("LZ_NORTH")
        assert label_for("LZ_WEST").startswith("West Texas")

    def test_hubs_are_labelled_as_hubs(self):
        for code in HUBS:
            assert "hub" in label_for(code).lower()

    def test_panhandle_names_amarillo(self):
        assert "Panhandle" in label_for("HB_PAN")
        assert "Amarillo" in label_for("HB_PAN")

    def test_an_unknown_code_degrades_to_itself(self):
        assert label_for("LZ_NOWHERE") == "LZ_NOWHERE"


class TestNameFor:
    def test_it_drops_the_code(self):
        from wattson.zones import name_for

        assert name_for("LZ_AEN") == "Austin Energy (Austin / Travis Co.)"
        assert "LZ_" not in name_for("LZ_RAYBN")

    def test_every_zone_and_hub_has_a_distinct_name(self):
        from wattson.zones import name_for

        names = [name_for(c) for c in LOAD_ZONES + HUBS]
        assert len(set(names)) == len(names)

    def test_unknown_codes_fall_back_to_the_code(self):
        from wattson.zones import name_for

        assert name_for("LZ_NOWHERE") == "LZ_NOWHERE"


class TestBaseCores:
    def test_one_core_is_base_core(self):
        from wattson.config import BASE_CORE, base_cores

        assert base_cores(1) is BASE_CORE

    def test_cores_add_up(self):
        from wattson.config import base_cores

        three = base_cores(3)
        assert (three.capacity_kwh, three.power_kw) == (117.6, 33.0)
        assert three.round_trip_efficiency == base_cores(1).round_trip_efficiency

    @pytest.mark.parametrize("bad", [0, 11])
    def test_out_of_range_counts_are_rejected(self, bad):
        from wattson.config import base_cores

        with pytest.raises(ValueError):
            base_cores(bad)
