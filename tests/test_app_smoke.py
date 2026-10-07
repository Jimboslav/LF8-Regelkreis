"""Regressionstests für Praxisauswahl und sichere Simulation."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"
CORE_PRESETS = {
    "RLT / Lüftung": {"Zulufttemperatur": 16.0, "VAV-Volumenstrom": 3000.0},
    "Heizung": {"Heizkreis-Vorlauftemperatur": 20.0, "Trinkwarmwasserbereitung": 45.0},
    "Kälte": {"Kaltwasser-Vorlauftemperatur": 6.0},
    "Wasser": {"Druckhaltung": 4.0, "Durchfluss": 18.0},
    "Druckluft": {"Netzdruck": 7.0},
    "Elektroantrieb": {"Drehzahl": 1200.0, "Position": 250.0},
}


class AppSmokeTests(unittest.TestCase):
    def test_all_core_presets_transfer_current_values(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.button[1].click().run()  # Physikalischer Wirkplan-Builder
        for category, variants in CORE_PRESETS.items():
            app.selectbox(key="wirkplan_anlagenart").set_value(category).run()
            self.assertFalse(app.exception, category)
            self.assertEqual(list(app.selectbox(key="wirkplan_regelungsvariante").options), list(variants))
            for variant, expected_setpoint in variants.items():
                app.selectbox(key="wirkplan_regelungsvariante").set_value(variant).run()
                self.assertFalse(app.button(key="wirkplan_sync_all").disabled, variant)
                app.button(key="wirkplan_sync_all").click().run()
                self.assertFalse(app.exception, variant)
                self.assertEqual(app.session_state["defaults"]["setpoint"], expected_setpoint, variant)

    def test_unreachable_temperature_blocks_transfer(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.button[1].click().run()
        app.number_input(key="wirkplan_temp_soll_c").set_value(60.0).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.button(key="wirkplan_sync_all").disabled)
        self.assertTrue(any("Erreichbarkeit" in item.value for item in app.error))

    def test_excessive_point_count_stops_safely(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.button(key="FormSubmitter:start_formular-Regelkreis erstellen").click().run()
        app.number_input(key="sim_t_end").set_value(1_000_000.0).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Simulationspunkte" in item.value for item in app.error))

    def test_position_preset_does_not_overshoot(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.button[1].click().run()
        app.selectbox(key="wirkplan_anlagenart").set_value("Elektroantrieb").run()
        app.selectbox(key="wirkplan_regelungsvariante").set_value("Position").run()
        app.button(key="wirkplan_sync_all").click().run()
        app.button[0].click().run()  # Simulation
        self.assertFalse(app.exception)
        overshoot = next(item for item in app.metric if item.label == "Überschwingen")
        self.assertLess(float(overshoot.value.replace(" %", "")), 5.0)

    def test_physical_reset_restores_default_variant_and_values(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.button[1].click().run()
        app.selectbox(key="wirkplan_anlagenart").set_value("Wasser").run()
        app.selectbox(key="wirkplan_regelungsvariante").set_value("Druckhaltung").run()
        app.button(key="wirkplan_reset_all").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key="wirkplan_anlagenart").value, "RLT / Lüftung")
        self.assertEqual(app.selectbox(key="wirkplan_regelungsvariante").value, "Zulufttemperatur")
        self.assertEqual(app.number_input(key="wirkplan_temp_soll_c").value, 21.0)


if __name__ == "__main__":
    unittest.main()
