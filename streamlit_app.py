import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState


def _praxis_variant(
    model, controlled, process, storage, disturbance, actuators, strategies,
    inertia="mittel", overshoot="Nein", offset="Nein", disturbances="Ja",
):
    """Kompakte Definition einer praxisnahen Regelungsvariante."""
    return {
        "model": model,
        "controlled": controlled,
        "process": process,
        "storage": storage,
        "disturbance": disturbance,
        "actuators": actuators,
        "strategies": strategies,
        "inertia": inertia,
        "overshoot": overshoot,
        "offset": offset,
        "disturbances": disturbances,
    }


_PI = ["Automatische Empfehlung", "P", "PI", "PID"]
_TEMP = ["Automatische Empfehlung", "PI", "PID", "Kaskade", "Zweipunkt"]
_STAGES = ["Automatische Empfehlung", "PI", "Stufen-/Kaskadensteuerung"]
_AIR = ["Ventilator ohne FU", "Ventilator mit FU", "EC-Ventilator", "VAV-Klappe"]
_PUMP = ["Pumpe ohne FU", "Pumpe mit FU", "EC-Pumpe", "Regelventil"]
_VALVE = ["2-Wege-Regelventil", "3-Wege-Mischventil", "Motorventil", "Magnetventil"]
_MOTOR = ["Motor ohne FU", "Motor mit FU", "EC-Motor", "Servoantrieb"]


PRACTICAL_PROCESS_CATALOG = {
    "RLT / Lüftung": {
        "Zulufttemperatur": _praxis_variant("Temperaturregelung", "Zulufttemperatur [°C]", "Heiz-/Kühlregister", "Luft- und Registermasse", "Außenluft / Last", _VALVE, _TEMP, "mittel"),
        "Raum- oder Ablufttemperatur": _praxis_variant("Temperaturregelung", "Raumtemperatur [°C]", "RLT-Anlage / Raum", "Gebäudemasse", "Außentemperatur / interne Last", _VALVE + _AIR, _TEMP, "sehr träge"),
        "Kanal-Differenzdruck": _praxis_variant("Generische Prozessstrecke", "Differenzdruck [Pa]", "Ventilator / Kanalnetz", "kompressibles Luftvolumen", "Klappen- und VAV-Stellung", _AIR, _PI, "schnell"),
        "Volumenstrom": _praxis_variant("Durchflussregelung", "Luftvolumenstrom [m³/h]", "Ventilator / Kanal", "Kanalvolumen", "Filterverschmutzung / Klappen", _AIR, _PI, "schnell"),
        "CO₂ / Luftqualität": _praxis_variant("Generische Prozessstrecke", "CO₂-Konzentration [ppm]", "Außenluftzufuhr / Raum", "Raumluftvolumen", "Personenbelegung", _AIR, ["Automatische Empfehlung", "PI", "Kaskade"], "sehr träge"),
        "Raum- oder Zuluftfeuchte": _praxis_variant("Generische Prozessstrecke", "relative Feuchte [% r. F.]", "Befeuchter / Entfeuchter", "Feuchtespeicherung", "Außenluft / Feuchtelast", ["Dampfbefeuchter", "Sprühbefeuchter", "Kühlregister", "Regelventil"], _TEMP, "träge"),
        "Mischlufttemperatur": _praxis_variant("Temperaturregelung", "Mischlufttemperatur [°C]", "Außen-/Umluftklappen", "Kanal- und Sensormasse", "Außenlufttemperatur", ["gekoppelte Mischluftklappen", "Einzelklappenantriebe"], ["Automatische Empfehlung", "PI", "Split-Range"], "mittel"),
        "Frostschutz": _praxis_variant("Temperaturregelung", "Temperatur nach Heizregister [°C]", "Vorheizregister", "Registermasse", "Frost / Luftstromausfall", _VALVE, ["Automatische Empfehlung", "PI", "Zweipunkt", "Sicherheitsbegrenzung"], "schnell", "Nein"),
    },
    "Heizung": {
        "witterungsgeführter Heizkreis": _praxis_variant("Temperaturregelung", "Vorlauftemperatur [°C]", "Mischer / Heizkreis", "Wasser- und Gebäudemasse", "Außentemperatur / Abnahme", _VALVE + _PUMP, ["Automatische Empfehlung", "PI", "Heizkurve + PI"], "träge"),
        "Heizkreis-Vorlauftemperatur": _praxis_variant("Temperaturregelung", "Vorlauftemperatur [°C]", "Wärmeerzeuger / Mischer", "Wasserinhalt", "Rücklauftemperatur", _VALVE, _TEMP, "träge"),
        "Rücklauftemperaturbegrenzung": _praxis_variant("Temperaturregelung", "Rücklauftemperatur [°C]", "Bypass / Mischer", "Wasserinhalt", "Wärmeabnahme", _VALVE, ["Automatische Empfehlung", "PI", "Begrenzungsregelung"], "träge"),
        "Raumtemperatur": _praxis_variant("Temperaturregelung", "Raumtemperatur [°C]", "Heizfläche / Raum", "Gebäudemasse", "Außentemperatur / Fremdwärme", _VALVE, _TEMP, "sehr träge"),
        "Kesseltemperatur": _praxis_variant("Temperaturregelung", "Kesseltemperatur [°C]", "Brenner / Kessel", "Kesselwasser und Metall", "Wärmeabnahme", ["modulierender Brenner", "mehrstufiger Brenner", "Elektroheizung"], _TEMP, "träge"),
        "Pufferspeicher-Ladung": _praxis_variant("Temperaturregelung", "Speichertemperatur [°C]", "Ladekreis", "Pufferspeicher", "Entnahme / Schichtung", _PUMP + _VALVE, ["Automatische Empfehlung", "PI", "Zweipunkt", "Kaskade"], "sehr träge"),
        "Trinkwarmwasserbereitung": _praxis_variant("Temperaturregelung", "Warmwassertemperatur [°C]", "Wärmetauscher / Speicher", "Warmwasservolumen", "Zapfung / Kaltwasser", _VALVE + _PUMP, _TEMP, "träge"),
    },
    "Kälte": {
        "Kaltwasser-Vorlauftemperatur": _praxis_variant("Temperaturregelung", "Kaltwasser-Vorlauf [°C]", "Kältemaschine / Verdampfer", "Wasserinhalt", "Kühllast", ["Verdichter mit FU", "Verdichterstufen", "Regelventil"], _TEMP, "träge"),
        "Kaltwasser-Rücklauftemperatur": _praxis_variant("Temperaturregelung", "Kaltwasser-Rücklauf [°C]", "Verbrauchernetz", "Wasserinhalt", "Kühllast", _PUMP, _TEMP, "träge"),
        "Kühlraumtemperatur": _praxis_variant("Temperaturregelung", "Raumtemperatur [°C]", "Verdampfer / Kühlraum", "Produkt- und Gebäudemasse", "Türöffnung / Einlagerung", ["Verdichter", "Magnetventil", "elektronisches Expansionsventil"], ["Automatische Empfehlung", "PI", "Zweipunkt"], "sehr träge"),
        "Verdampfungsdruck": _praxis_variant("Druckregelung", "Verdampfungsdruck [bar]", "Verdichter / Verdampfer", "Kältemittelfüllung", "Kühllast", ["Verdichter mit FU", "Verdichterstufen", "Saugdruckregler"], _STAGES, "schnell"),
        "Verflüssigungsdruck": _praxis_variant("Druckregelung", "Verflüssigungsdruck [bar]", "Verflüssiger", "Kältemittelfüllung", "Außentemperatur", ["Verflüssigerlüfter mit FU", "EC-Lüfter", "Wasserventil"], _PI, "mittel"),
        "Überhitzungsregelung": _praxis_variant("Generische Prozessstrecke", "Überhitzung [K]", "Verdampfer / Expansionsventil", "Kältemittelfüllung", "Last- und Druckänderung", ["elektronisches Expansionsventil", "thermostatisches Expansionsventil"], ["Automatische Empfehlung", "PI", "PID"], "schnell"),
        "Kältespeicher-Ladung": _praxis_variant("Temperaturregelung", "Speichertemperatur [°C]", "Ladekreis", "Kältespeicher", "Kälteentnahme", _PUMP + _VALVE, ["Automatische Empfehlung", "PI", "Zweipunkt"], "sehr träge"),
    },
    "Wasser": {
        "Behälter-Füllstand": _praxis_variant("Füllstandsregelung", "Füllstand [m]", "Zulauf / Behälter", "Behältervolumen", "Abfluss", _PUMP + _VALVE, _PI, "träge"),
        "Druckhaltung": _praxis_variant("Druckregelung", "Netzdruck [bar]", "Pumpe / Rohrnetz", "Druckbehälter", "Verbrauch", _PUMP, _PI, "mittel"),
        "Durchfluss": _praxis_variant("Durchflussregelung", "Wasserdurchfluss [m³/h]", "Pumpe / Ventil / Rohr", "Rohrvolumen", "Vordruck / Verbraucher", _PUMP, _PI, "schnell"),
        "Pumpenkaskade": _praxis_variant("Druckregelung", "Netzdruck [bar]", "Mehrpumpenanlage", "Druckbehälter", "Verbrauch", ["Pumpen ohne FU", "Führungspumpe mit FU", "alle Pumpen mit FU"], _STAGES, "mittel"),
        "Brunnen- oder Hochbehälter": _praxis_variant("Füllstandsregelung", "Wasserstand [m]", "Förderpumpe / Speicher", "Brunnen oder Hochbehälter", "Entnahme / Zulauf", _PUMP, ["Automatische Empfehlung", "PI", "Zweipunkt"], "sehr träge"),
        "Wassertemperatur": _praxis_variant("Temperaturregelung", "Wassertemperatur [°C]", "Wärmetauscher", "Wasservolumen", "Zulauftemperatur / Durchfluss", _VALVE, _TEMP, "träge"),
    },
    "Abwasser": {
        "Pumpensumpf-Füllstand": _praxis_variant("Füllstandsregelung", "Füllstand [m]", "Pumpensumpf / Pumpen", "Sumpfvolumen", "schwankender Zulauf", ["Pumpe ohne FU", "Pumpe mit FU", "Mehrpumpenkaskade"], ["Automatische Empfehlung", "Zweipunkt", "Stufen-/Kaskadensteuerung", "PI"], "träge"),
        "Zulauf- oder Ablaufmenge": _praxis_variant("Durchflussregelung", "Durchfluss [m³/h]", "Pumpe / Gerinne", "Becken- und Rohrvolumen", "Zulaufschwankung", _PUMP, _PI, "mittel"),
        "Sauerstoff im Belebungsbecken": _praxis_variant("Generische Prozessstrecke", "Sauerstoff [mg/l]", "Belüftung / Becken", "Beckenvolumen und Biomasse", "Schmutzfracht", ["Gebläse mit FU", "EC-Gebläse", "Belüfterventil"], ["Automatische Empfehlung", "PI", "Kaskade"], "sehr träge"),
        "pH-Wert": _praxis_variant("Generische Prozessstrecke", "pH-Wert", "Neutralisation / Becken", "Beckenvolumen", "Zulauf-pH und Pufferkapazität", ["Dosierpumpe Säure", "Dosierpumpe Lauge", "Split-Range-Dosierung"], ["Automatische Empfehlung", "PI", "Split-Range"], "träge", "Nein"),
        "Leitfähigkeit": _praxis_variant("Generische Prozessstrecke", "Leitfähigkeit [µS/cm]", "Dosierung / Spülung", "Beckenvolumen", "Stoffeintrag", ["Dosierpumpe", "Spülventil"], _PI, "träge"),
        "Chemikaliendosierung": _praxis_variant("Durchflussregelung", "Dosierstrom [l/h]", "Dosierpumpe / Leitung", "Leitungsvolumen", "Gegendruck / Konzentration", ["Membrandosierpumpe", "Schlauchpumpe", "Regelventil"], _PI, "schnell"),
    },
    "Druckluft": {
        "Netzdruck": _praxis_variant("Druckregelung", "Netzdruck [bar]", "Verdichter / Netz", "Druckluftbehälter", "Luftverbrauch", ["Verdichter mit FU", "Last-Leerlauf-Verdichter", "Verdichterkaskade"], _STAGES, "mittel"),
        "Behälterdruck": _praxis_variant("Druckregelung", "Behälterdruck [bar]", "Verdichter / Behälter", "Druckbehälter", "Entnahme", ["Verdichter ohne FU", "Verdichter mit FU", "Einlassventil"], ["Automatische Empfehlung", "PI", "Zweipunkt"], "mittel"),
        "Verdichterkaskade": _praxis_variant("Druckregelung", "Netzdruck [bar]", "Mehrverdichteranlage", "Netz- und Behältervolumen", "Verbrauch", ["Grundlast-/Spitzenlastverdichter", "alle Verdichter mit FU"], _STAGES, "mittel"),
        "Taupunkt": _praxis_variant("Generische Prozessstrecke", "Drucktaupunkt [°C]", "Trockner", "Trocknermasse / Adsorber", "Feuchtelast", ["Kältetrockner", "Adsorptionstrockner", "Bypassventil"], ["Automatische Empfehlung", "PI", "Zweipunkt"], "sehr träge"),
    },
    "Elektroantrieb": {
        "Drehzahl": _praxis_variant("Drehzahlregelung", "Drehzahl [1/min]", "Motor / Last", "Trägheitsmoment", "Lastmoment", _MOTOR, _PI, "schnell"),
        "Position": _praxis_variant("Position / Mechanik", "Position [mm]", "Antrieb / Mechanik", "Masse / Feder", "Lastkraft / Reibung", ["Servoantrieb", "Schrittmotor", "Motor mit FU und Geber", "Linearantrieb"], ["Automatische Empfehlung", "P", "PI", "PID"], "mittel"),
        "Drehmoment": _praxis_variant("Generische Prozessstrecke", "Drehmoment [Nm]", "Motor / Last", "mechanische Trägheit", "Lastmoment", ["Motor mit FU", "Servoantrieb", "DC-Antrieb"], ["Automatische Empfehlung", "PI", "PID"], "schnell"),
        "Bandgeschwindigkeit": _praxis_variant("Drehzahlregelung", "Bandgeschwindigkeit [m/s]", "Motor / Getriebe / Band", "Massen und Trägheit", "Beladung / Schlupf", _MOTOR, _PI, "mittel"),
        "Gleichlauf / Synchronisation": _praxis_variant("Drehzahlregelung", "Drehzahl- oder Positionsabweichung", "gekoppelte Antriebe", "Trägheitsmomente", "Lastunterschiede", ["mehrere Servoantriebe", "mehrere FU-Antriebe", "elektronische Königswelle"], ["Automatische Empfehlung", "PI", "PID", "Kaskade"], "schnell"),
    },
    "Raumautomation": {
        "Raumtemperatur Heizen": _praxis_variant("Temperaturregelung", "Raumtemperatur [°C]", "Heizfläche / Raum", "Gebäudemasse", "Außentemperatur / Belegung", _VALVE, _TEMP, "sehr träge"),
        "Raumtemperatur Kühlen": _praxis_variant("Temperaturregelung", "Raumtemperatur [°C]", "Kühldecke / Raum", "Gebäudemasse", "Außentemperatur / solare Last", _VALVE, _TEMP, "sehr träge"),
        "Heiz-/Kühlsequenz": _praxis_variant("Temperaturregelung", "Raumtemperatur [°C]", "Heiz- und Kühlventil", "Gebäudemasse", "Wärme- und Kühllast", ["Heizventil + Kühlventil", "6-Wege-Ventil", "Fan-Coil"], ["Automatische Empfehlung", "Split-Range", "PI"], "sehr träge"),
        "CO₂-geführte Lüftung": _praxis_variant("Generische Prozessstrecke", "CO₂ [ppm]", "Luftwechsel / Raum", "Raumluftvolumen", "Belegung", _AIR, ["Automatische Empfehlung", "PI", "Kaskade"], "sehr träge"),
        "VAV-Volumenstrom": _praxis_variant("Durchflussregelung", "Volumenstrom [m³/h]", "VAV-Box / Kanal", "Kanalvolumen", "Kanaldruck", ["VAV-Klappe", "EC-Ventilator"], _PI, "schnell"),
        "Raumfeuchte": _praxis_variant("Generische Prozessstrecke", "relative Feuchte [% r. F.]", "Befeuchter / Entfeuchter", "Raum- und Materialfeuchte", "Personen / Außenluft", ["Raumbefeuchter", "Kühlventil", "Luftmengensteller"], _TEMP, "träge"),
    },
    "Dampf": {
        "Dampfdruck": _praxis_variant("Druckregelung", "Dampfdruck [bar]", "Dampferzeuger", "Kessel- und Dampfvolumen", "Dampfentnahme", ["modulierender Brenner", "Elektroheizung", "Druckregelventil"], _TEMP, "träge"),
        "Dampftemperatur": _praxis_variant("Temperaturregelung", "Dampftemperatur [°C]", "Überhitzer / Einspritzung", "Rohr- und Metallmasse", "Dampfmenge", ["Einspritzventil", "Brennerleistung"], _TEMP, "träge"),
        "Kesselwasserstand": _praxis_variant("Füllstandsregelung", "Kesselwasserstand", "Speisewasser / Trommel", "Trommelvolumen", "Dampfentnahme", ["Speisewasserventil", "Speisewasserpumpe mit FU"], ["Automatische Empfehlung", "PI", "Kaskade", "Dreipunktregelung"], "mittel"),
        "Kondensatstand": _praxis_variant("Füllstandsregelung", "Kondensatstand", "Kondensatbehälter", "Behältervolumen", "Kondensatanfall", _PUMP + _VALVE, ["Automatische Empfehlung", "PI", "Zweipunkt"], "träge"),
    },
    "Prozesswärme": {
        "Ofentemperatur": _praxis_variant("Temperaturregelung", "Ofentemperatur [°C]", "Brenner / Heizelement", "Ofen- und Produktmasse", "Beschickung / Türöffnung", ["modulierender Brenner", "Thyristorsteller", "Schützstufen"], _TEMP, "sehr träge"),
        "Zonentemperatur": _praxis_variant("Temperaturregelung", "Zonentemperatur [°C]", "Heizzone", "Zonen- und Produktmasse", "Nachbarzonen / Produkt", ["Thyristorsteller", "Heizregister", "Brennerzone"], ["Automatische Empfehlung", "PI", "PID", "Kaskade"], "träge"),
        "Wärmeträgertemperatur": _praxis_variant("Temperaturregelung", "Wärmeträgertemperatur [°C]", "Erhitzer / Kreislauf", "Fluid- und Anlagenmasse", "Prozessabnahme", _VALVE + _PUMP, _TEMP, "träge"),
        "Kaskade Produkt/Medium": _praxis_variant("Temperaturregelung", "Produkttemperatur [°C]", "Medium und Produkt", "Produktmasse", "Durchsatz / Eintrittstemperatur", _VALVE, ["Automatische Empfehlung", "Kaskade", "PI", "PID"], "sehr träge"),
    },
    "Dosierung / Chemie": {
        "pH-Wert": _praxis_variant("Generische Prozessstrecke", "pH-Wert", "Reaktor / Neutralisation", "Reaktorvolumen", "Zulauf-pH / Pufferkapazität", ["Säure-Dosierpumpe", "Lauge-Dosierpumpe", "Split-Range-Dosierung"], ["Automatische Empfehlung", "PI", "Split-Range"], "träge"),
        "Leitfähigkeit": _praxis_variant("Generische Prozessstrecke", "Leitfähigkeit [µS/cm]", "Dosierung / Spülung", "Prozessvolumen", "Salz- oder Chemikalieneintrag", ["Dosierpumpe", "Spülventil"], _PI, "träge"),
        "Konzentration": _praxis_variant("Generische Prozessstrecke", "Konzentration [%]", "Mischer / Reaktor", "Reaktorvolumen", "Zulaufkonzentration", ["Dosierpumpe", "Regelventil", "Mischventil"], _PI, "träge"),
        "Mischungsverhältnis": _praxis_variant("Durchflussregelung", "Mischungsverhältnis", "zwei Stoffströme / Mischer", "Mischvolumen", "Vordruck / Stoffeigenschaften", ["zwei Regelventile", "zwei Dosierpumpen"], ["Automatische Empfehlung", "Verhältnisregelung", "Kaskade", "PI"], "mittel"),
        "Dosiermenge": _praxis_variant("Durchflussregelung", "Dosierstrom [l/h]", "Dosierpumpe", "Leitungsvolumen", "Gegendruck", ["Membrandosierpumpe", "Schlauchpumpe", "Schneckenförderer"], _PI, "schnell"),
    },
    "Hydraulik / Pneumatik": {
        "Systemdruck": _praxis_variant("Druckregelung", "Druck [bar]", "Pumpe / Kompressor / Ventil", "Speicher und Leitungsvolumen", "Last / Leckage", ["Pumpe mit FU", "Proportionalventil", "Druckregelventil"], _PI, "schnell"),
        "Zylinderposition": _praxis_variant("Position / Mechanik", "Position [mm]", "Zylinder / Last", "Masse und Fluidkompressibilität", "Lastkraft / Reibung", ["Proportionalventil", "Servoventil", "Pneumatikventil"], ["Automatische Empfehlung", "P", "PI", "PID"], "schnell"),
        "Kraft": _praxis_variant("Generische Prozessstrecke", "Kraft [N]", "Zylinder / Werkzeug", "mechanische Nachgiebigkeit", "Gegenkraft", ["Proportional-Druckventil", "Servoventil"], ["Automatische Empfehlung", "PI", "PID"], "schnell"),
        "Geschwindigkeit": _praxis_variant("Durchflussregelung", "Geschwindigkeit [mm/s]", "Ventil / Zylinder", "bewegte Masse", "Last / Reibung", ["Proportionalventil", "Stromregelventil", "Servoventil"], _PI, "schnell"),
    },
    "Energie": {
        "Leistungsbegrenzung": _praxis_variant("Generische Prozessstrecke", "Bezugsleistung [kW]", "Verbraucher / Leistungssteller", "thermische und elektrische Flexibilität", "Lastsprünge", ["Leistungssollwert", "Lastabwurf", "Batteriewechselrichter"], ["Automatische Empfehlung", "PI", "Prioritätssteuerung"], "mittel"),
        "Eigenverbrauchsoptimierung": _praxis_variant("Generische Prozessstrecke", "Netzleistung [kW]", "PV / Speicher / Verbraucher", "Batteriespeicher", "PV-Erzeugung / Verbrauch", ["Batteriewechselrichter", "steuerbare Verbraucher", "Wärmepumpe"], ["Automatische Empfehlung", "PI", "Energiemanagement"], "mittel"),
        "Speicherladung": _praxis_variant("Generische Prozessstrecke", "Ladezustand [%]", "Batterie / Ladegerät", "Batteriekapazität", "Verbrauch / Erzeugung", ["Batteriewechselrichter", "Ladegerät"], ["Automatische Empfehlung", "Leistungsregelung", "Energiemanagement"], "sehr träge"),
        "Lastmanagement": _praxis_variant("Generische Prozessstrecke", "Gesamtleistung [kW]", "Verbrauchergruppen", "verschiebbare Lasten", "Produktions- und Belegungsplan", ["Lastfreigaben", "Sollwertvorgaben", "Lastabwurf"], ["Automatische Empfehlung", "Prioritätssteuerung", "Energiemanagement"], "mittel"),
    },
}




# Regelkreis-Labor 2.0 – korrigierter Prüfstand, 26.09.2026
# Alle Modelle verwenden Sekunden; Stellgrößen sind Prozentpunkte.
# Abhängigkeiten: streamlit, streamlit-flow-component, numpy, pandas,
# matplotlib, scipy (siehe requirements.txt).
import copy
import hashlib
import html
import json
import math
from collections import deque
from scipy.linalg import expm

APP_VERSION = "2.0.0"
MAX_POINTS = 100_001
MAX_INTERNAL_STEPS = 500_000
SUPPORTED_STRATEGIES = {
    "Automatische Empfehlung", "P", "PI", "PID", "Zweipunkt",
    "Stufen-/Kaskadensteuerung",
}


class ModelError(ValueError):
    """Verständlicher Eingabe-/Modellfehler vor einer Allokation."""


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def default_config():
    return dict(
        version=APP_VERSION, title="Freies Regelkreis-Labor", kind="PT1",
        category="", variant="", concept=False,
        model_note="Frei parametrierbare lineare Lehrstrecke; Werte sind keine Anlagenauslegung.",
        assumptions=[], actuator="Idealer stetiger Aktor", strategy="Automatische Empfehlung",
        controller="PI", automatic=True, kp=1.0, ki=0.5, kd=0.0, beta=0.0,
        derivative_filter=0.05, hysteresis=0.05, no_overshoot=True,
        overshoot_limit=0.5, tolerance_percent=2.0,
        ks=1.0, ts=2.0, zeta=0.7, omega0=2.0,
        unit="Einheit", y0=0.0, baseline=0.0, setpoint=1.0,
        sensor_min=-10.0, sensor_max=10.0, sensor_tau=0.0,
        noise=0.0, seed=42, delay=0.0, actuator_tau=0.0, rate=10000.0,
        actuator_mode="Stetig", stages=3, u_min=-100.0, u_max=100.0,
        bias=0.0, t_end=30.0, dt=0.01, setpoint_ramp=0.0,
        disturbance_position="Keine Störung", disturbance_time=10.0,
        disturbance_value=0.0, disturbance_unit="Einheit/s",
        boundary_min=None, boundary_max=None,
        temp_medium="Wasser", temp_volume=0.5, temp_power=30.0,
        temp_efficiency=0.95, temp_loss=450.0, temp_ambient=20.0,
        temp_rho=998.0, temp_cp=4180.0, temp_extra_capacity=0.0,
        temp_flow=0.0, temp_inlet=20.0, temp_mode="Heizen",
        tank_volume=5.0, tank_height=2.5, tank_qmax=12.0, tank_baseflow=6.0,
        tank_mode="Zulauf regeln", tank_shape="Aufrechter Zylinder",
        motor_power=7.5, motor_rpm=1450.0, motor_sync_rpm=1500.0,
        motor_inertia=0.25, motor_load=30.0, motor_friction=0.01,
        motor_ramp=4.0, motor_eta=0.9, motor_cosphi=0.82,
        motor_voltage=400.0, motor_length=50.0, motor_section=2.5,
        motor_cable_temp=70.0, motor_reactance=0.08,
        motor_start_ratio=1.8, motor_breakdown_ratio=2.5,
        motor_drive="FU: Momentenbetrieb", motor_gear=1.0, motor_diameter=0.2,
        motor_band=False,
        gas_volume=2.0, gas_qmax=180.0, gas_demand=90.0,
        gas_pmax=10.0, gas_pref=1.01325, gas_tref=293.15, gas_temperature=293.15,
        gas_ambient=1.01325,
        flow_qref=100.0, flow_length=25.0, flow_diameter=80.0,
        flow_friction=0.025, flow_density=998.0, flow_valve_dp=1.5,
        flow_supply_dp=3.0, flow_tau=1.0, flow_scale=1.0,
        mech_mass=40.0, mech_spring=1500.0, mech_damping=220.0,
        mech_force=2000.0, mech_load=0.0, mech_hub=500.0, mech_v0=0.0,
    )


def motor_data(c):
    torque = c["motor_power"] * 1000.0 / (c["motor_rpm"] * 2 * np.pi / 60)
    current = c["motor_power"] * 1000 / (np.sqrt(3) * c["motor_voltage"] *
                                           c["motor_eta"] * c["motor_cosphi"])
    rho = 0.0178 * (1 + 0.00393 * (c["motor_cable_temp"] - 20))
    resistance = rho * c["motor_length"] / c["motor_section"]
    reactance = c["motor_reactance"] * c["motor_length"] / 1000
    drop = np.sqrt(3) * current * (resistance * c["motor_cosphi"] +
            reactance * np.sqrt(1 - c["motor_cosphi"]**2))
    scale = np.pi*c["motor_diameter"]/(60*c["motor_gear"]) if c["motor_band"] else 1.0
    return dict(torque=torque, current=current, drop=drop,
                cable_loss=3*current**2*resistance, speed_scale=scale)


def flow_data(c, opening=100.0, disturbance=0.0):
    """Quadratische Rohr-/Ventilwiderstände, konstanter verfügbarer Differenzdruck.

    Keine Verwechslung von Stoffaufenthaltszeit und Durchflusszeitkonstante.
    qref und Q sind m³/h; Drücke bar; Innendurchmesser mm.
    """
    diameter = c["flow_diameter"] / 1000
    area = np.pi * diameter**2 / 4
    velocity = c["flow_qref"] / (3600 * area)
    pipe_dp = c["flow_density"] * c["flow_friction"] * c["flow_length"] / diameter * velocity**2 / 2e5
    fraction = max(0.0, opening) / 100
    available = max(0.0, c["flow_supply_dp"] + disturbance)
    q = 0.0 if fraction == 0 else c["flow_qref"] * np.sqrt(
        available / (pipe_dp + c["flow_valve_dp"] / fraction**2))
    return dict(q=q*c["flow_scale"], pipe_dp=pipe_dp,
                volume=area*c["flow_length"], velocity=velocity,
                residence=area*c["flow_length"]*3600/c["flow_qref"])


def physical_model(c):
    """Affine kontinuierliche Bilanz: x'=Ax+Bu+f+Dd, y=offset+C x.

    Nichtlineare Motor-/Durchflussmodelle werden im Simulator separat behandelt.
    A/B werden dort ausschließlich für die lokale Startauslegung verwendet.
    """
    k = c["kind"]
    unit = c["unit"]
    offset = c["baseline"]
    A = np.array([[-1.0/c["ts"]]])
    B = np.array([c["ks"]/c["ts"]])
    D = np.array([1.0])
    f = np.zeros(1)
    x0 = np.array([c["y0"]-offset])
    metrics = {}
    lower, upper = c["boundary_min"], c["boundary_max"]
    note = c["model_note"]
    if k == "PT2":
        A=np.array([[0.,1.],[-c["omega0"]**2,-2*c["zeta"]*c["omega0"]]])
        B=np.array([0.,c["ks"]*c["omega0"]**2]);D=np.array([0.,1.]);f=np.zeros(2)
        x0=np.array([c["y0"]-offset,0.])
    elif k == "thermal":
        capacity=c["temp_rho"]*c["temp_volume"]*c["temp_cp"]+c["temp_extra_capacity"]
        transport=c["temp_rho"]*c["temp_cp"]*c["temp_flow"]/3600
        H=c["temp_loss"]+transport
        direction=1 if c["temp_mode"]=="Heizen" else -1
        offset=(c["temp_loss"]*c["temp_ambient"]+transport*c["temp_inlet"])/H
        A=np.array([[-H/capacity]])
        B=np.array([direction*c["temp_power"]*1000*c["temp_efficiency"]/(100*capacity)])
        D=np.array([1/capacity]);x0=np.array([c["y0"]-offset])
        metrics=dict(Waermekapazitaet_J_K=capacity,Zeitkonstante_s=capacity/H,
                     Passive_Gleichgewichtstemperatur_C=offset,
                     Endtemperatur_100_Prozent_C=offset-B[0]/A[0,0]*100)
    elif k == "tank":
        area=c["tank_volume"]/c["tank_height"]
        sign=1 if c["tank_mode"]=="Zulauf regeln" else -1
        offset=0.;A=np.zeros((1,1));B=np.array([sign*c["tank_qmax"]/(360000*area)])
        f=np.array([-sign*c["tank_baseflow"]/(3600*area)])
        D=np.array([-sign/(3600*area)]);x0=np.array([c["y0"]])
        lower,upper=0.,c["tank_height"]
        net=c["tank_qmax"]-c["tank_baseflow"]
        fill=None if net<=0 else abs(c["setpoint"]-c["y0"])*area/net*3600
        metrics=dict(Querschnitt_m2=area,Stationaere_Stellgroesse_Prozent=100*c["tank_baseflow"]/c["tank_qmax"],
                     Zeit_bis_Soll_bei_Maximalleistung_s=fill)
    elif k == "gas":
        coefficient=c["gas_pref"]*c["gas_temperature"]/(c["gas_tref"]*c["gas_volume"]*3600)
        offset=0.;A=np.zeros((1,1));B=np.array([coefficient*c["gas_qmax"]/100])
        D=np.array([-coefficient]);f=np.array([-coefficient*c["gas_demand"]]);x0=np.array([c["y0"]])
        lower,upper=0.,None
        net=c["gas_qmax"]-c["gas_demand"]
        fill=None if net<=0 else max(0.,c["setpoint"]-c["y0"])/(coefficient*net)
        metrics=dict(Zeit_bis_Solldruck_s=fill,Gasfaktor_bar_s_pro_m3h=coefficient,
                     Anfangsdruck_absolut_bar=c["gas_ambient"]+c["y0"],
                     Stationaere_Stellgroesse_Prozent=100*c["gas_demand"]/c["gas_qmax"])
    elif k == "mechanics":
        offset=0.;A=np.array([[0.,1.],[-c["mech_spring"]/c["mech_mass"],-c["mech_damping"]/c["mech_mass"]]])
        B=np.array([0.,10*c["mech_force"]/c["mech_mass"]])
        D=np.array([0.,-1000/c["mech_mass"]]);f=D*c["mech_load"]
        x0=np.array([c["y0"],c["mech_v0"]]);lower,upper=0.,c["mech_hub"]
        metrics=dict(Ks_mm_pro_Prozent=10*c["mech_force"]/c["mech_spring"],
                     Eigenkreisfrequenz_rad_s=np.sqrt(c["mech_spring"]/c["mech_mass"]),
                     Daempfungsgrad=c["mech_damping"]/(2*np.sqrt(c["mech_spring"]*c["mech_mass"])),
                     Kraft_bei_Soll_N=c["mech_spring"]*c["setpoint"]/1000+c["mech_load"])
    elif k == "motor":
        data=motor_data(c);scale=data["speed_scale"]*60/(2*np.pi)
        offset=0.;A=np.array([[-c["motor_friction"]/c["motor_inertia"]]])
        B=np.array([data["torque"]*scale/(100*c["motor_inertia"])])
        D=np.array([-scale/c["motor_inertia"]]);f=D*c["motor_load"]
        x0=np.array([c["y0"]]);lower,upper=0.,None
        metrics={"Nennmoment_Nm":data["torque"],"Nennstrom_A":data["current"],
                 "Spannungsfall_V":data["drop"],"Leitungsverluste_W":data["cable_loss"]}
    elif k == "flow":
        offset=0.;qmax=flow_data(c)["q"]
        A=np.array([[-1/c["flow_tau"]]])
        B=np.array([qmax/(100*c["flow_tau"])]);D=np.zeros(1);f=np.zeros(1)
        x0=np.array([c["y0"]]);lower,upper=0.,None
        metrics=flow_data(c);metrics["maximaler_Durchfluss"]=metrics.pop("q")
        metrics["identifizierte_Zeitkonstante_s"]=c["flow_tau"]
    C=np.zeros(len(B));C[0]=1.
    return dict(A=A,B=B,D=D,f=f,x0=x0,C=C,offset=offset,
                lower=lower,upper=upper,metrics=metrics,note=note,unit=unit)


def equilibrium_input(c,m,y):
    if c["kind"]=="flow":
        lo,hi=0.,100.
        for _ in range(40):
            mid=(lo+hi)/2
            if flow_data(c,mid)["q"]<y:lo=mid
            else:hi=mid
        return (lo+hi)/2
    j=int(np.argmax(np.abs(m["B"])))
    x=np.zeros(len(m["B"]));x[0]=y-m["offset"]
    return -(float(m["A"][j]@x)+m["f"][j])/m["B"][j]


def tune_controller(c):
    """Konservative Startwerte; Eignung wird im Ergebnis gegen Anforderungen geprüft."""
    d=copy.deepcopy(c)
    if d["strategy"]=="Stufen-/Kaskadensteuerung":d["actuator_mode"]="Stufen"
    m=physical_model(d)
    if not d["automatic"]:return d
    if d["strategy"] not in SUPPORTED_STRATEGIES:return d
    two=d["strategy"]=="Zweipunkt" or (d["strategy"]=="Automatische Empfehlung" and d["actuator_mode"]=="Ein/Aus")
    d["controller"]="Zweipunkt" if two else (d["strategy"] if d["strategy"] in {"P","PI","PID"} else "PI")
    A,B=m["A"],m["B"]
    slow=max(d["delay"]*4,d["sensor_tau"]*4,d["actuator_tau"]*4,d["dt"]*50,0.05)
    if len(B)==1 and abs(A[0,0])<1e-12:
        span=max(abs(d["setpoint"]-d["y0"]),abs(d["sensor_max"]-d["sensor_min"])*.1)
        lam=max(slow,span/(abs(B[0])*25))
        kp=2/(B[0]*lam);ki=1/(B[0]*lam**2);kd=0.
    elif len(B)==1:
        tau=-1/A[0,0];gain=-B[0]/A[0,0]
        lam=max(slow,tau)
        kp=tau/(gain*lam);ki=kp/tau;kd=0.
    else:
        w=np.sqrt(-A[1,0]);gain=B[1]/w**2
        # PID: low proportional gain, derivative increases physical damping.
        kp=.15/gain
        target_damping=2.4*w
        kd=max(0.,(target_damping+A[1,1])/B[1])
        ki=.02*w/gain
        if d["controller"]=="PI":
            ki=min(ki,max(0.,(-A[1,1])*(1+kp*gain)/(10*abs(gain))))
            if -A[1,1]<1e-9:
                d["controller"]="PID" if d["strategy"]=="Automatische Empfehlung" else "PI"
        if d["strategy"]=="Automatische Empfehlung":d["controller"]="PID"
    d.update(kp=float(kp),ki=float(ki),kd=float(kd),beta=0.)
    if d["controller"]=="P":d.update(ki=0.,kd=0.,beta=1.)
    elif d["controller"]=="PI":d["kd"]=0.
    d["bias"]=float(np.clip(equilibrium_input(d,m,d["y0"]),d["u_min"],d["u_max"]))
    return d


def loop_poles(c,m):
    """Kontinuierliche linearisierte Stabilität inkl. Messglied, D-Filter, Aktor-PT1.

    Totzeit, Sättigung und Stellrate bleiben ausdrücklich außerhalb dieser Prüfung.
    """
    if c["controller"]=="Zweipunkt":return np.array([])
    n=len(m["B"])
    # states plant[n], integral, derivative filter measurement, sensor, actuator
    def deriv(z):
        x=z[:n];integ,filtered,sensor,act=z[n:]
        y=float(m["C"]@x)
        ym=sensor if c["sensor_tau"]>0 else y
        kd=c["kd"] if c["controller"]=="PID" else 0.
        ki=c["ki"] if c["controller"] in {"PI","PID"} else 0.
        raw=-c["kp"]*ym+integ-kd*(ym-filtered)/c["derivative_filter"]
        u=act if c["actuator_tau"]>0 else raw
        return np.r_[m["A"]@x+m["B"]*u,
                     -ki*ym if ki else -integ,
                     (ym-filtered)/c["derivative_filter"],
                     (y-sensor)/c["sensor_tau"] if c["sensor_tau"]>0 else -sensor,
                     (raw-act)/c["actuator_tau"] if c["actuator_tau"]>0 else -act]
    eye=np.eye(n+4)
    return np.linalg.eigvals(np.column_stack([deriv(v) for v in eye]))


def validate_config(c):
    errors=[];warnings=[]
    if c["kind"] not in {"PT1","PT2","thermal","tank","gas","flow","motor","mechanics"}:
        return ["Unbekannter Modelltyp."],[]
    if c["controller"] not in {"P","PI","PID","Zweipunkt"}:errors.append("Unbekannter Reglertyp.")
    if c["actuator_mode"] not in {"Stetig","Ein/Aus","Stufen"}:errors.append("Unbekannte Ansteuerung.")
    if c["disturbance_position"] not in {"Keine Störung","Vor der Strecke","Prozessstörung","Am Ausgang"}:errors.append("Unbekannter Störungsort.")
    if not c["u_min"]<=c["bias"]<=c["u_max"] and not c["automatic"]:errors.append("Anfangsstellwert liegt außerhalb des Stellbereichs.")
    for key,val in c.items():
        if isinstance(val,(int,float)) and not isinstance(val,bool) and not np.isfinite(val):
            errors.append(f"{key}: nur endliche Zahlen sind zulässig.")
    if errors:return errors,warnings
    for key in ("ts","omega0","derivative_filter","rate","dt","t_end","temp_volume","temp_power",
                "temp_loss","temp_rho","temp_cp","tank_volume","tank_height","tank_qmax",
                "motor_power","motor_rpm","motor_sync_rpm","motor_inertia","motor_voltage","motor_section",
                "motor_gear","motor_diameter","gas_volume","gas_qmax","gas_pmax","gas_pref","gas_tref",
                "gas_temperature","gas_ambient","flow_qref","flow_diameter","flow_density","flow_valve_dp",
                "flow_supply_dp","flow_tau","flow_scale","mech_mass","mech_spring","mech_force","mech_hub"):
        if c[key]<=0:errors.append(f"{key} muss größer als null sein.")
    for key in ("sensor_tau","delay","actuator_tau","noise","zeta","temp_extra_capacity","temp_flow",
                "tank_baseflow","motor_load","motor_friction","motor_length","motor_reactance","motor_ramp",
                "gas_demand","flow_length","flow_friction","mech_damping","disturbance_time","setpoint_ramp"):
        if c[key]<0:errors.append(f"{key} darf nicht negativ sein.")
    for key in ("temp_efficiency","motor_eta","motor_cosphi"):
        if not 0<c[key]<=1:errors.append(f"{key}: zulässig ist 0 < Wert ≤ 1.")
    if not c["u_min"]<c["u_max"]:errors.append("Stellgröße min. muss kleiner als max. sein.")
    if not -100<=c["u_min"]<c["u_max"]<=100:errors.append("Stellbereich muss innerhalb −100 bis 100 % liegen.")
    if not c["sensor_min"]<c["sensor_max"]:errors.append("Messbereich min. muss kleiner als max. sein.")
    if not c["sensor_min"]<=c["setpoint"]<=c["sensor_max"]:errors.append("Sollwert liegt außerhalb des Messbereichs.")
    if not c["sensor_min"]<=c["y0"]<=c["sensor_max"]:errors.append("Anfangswert liegt außerhalb des Messbereichs.")
    if not 0<=c["beta"]<=1:errors.append("Sollwertgewicht β muss zwischen 0 und 1 liegen.")
    if c["seed"]<0 or int(c["seed"])!=c["seed"]:errors.append("Zufallsstartwert muss eine nichtnegative ganze Zahl sein.")
    if c["tolerance_percent"]<=0 or c["overshoot_limit"]<0:errors.append("Toleranz muss positiv und Überschwinggrenze nichtnegativ sein.")
    if c["motor_cable_temp"]<=-234:errors.append("Leitertemperatur außerhalb des Widerstandsmodells.")
    if c["hysteresis"]<=0:errors.append("Hysterese muss größer als null sein.")
    if c["stages"]<2 or int(c["stages"])!=c["stages"]:errors.append("Mindestens zwei ganze Stufen erforderlich.")
    if c["strategy"] not in SUPPORTED_STRATEGIES:
        errors.append(f"„{c['strategy']}“ ist eine Konzeptoption. Ein passendes Mehrkreis-/Logikmodell ist nicht implementiert. Bitte eine unterstützte Strategie wählen.")
    if c["dt"]>=c["t_end"]:errors.append("dt muss kleiner als die Simulationsdauer sein.")
    if c["dt"]>0 and c["t_end"]>0:
        if c["t_end"]/c["dt"]>MAX_POINTS-1:errors.append(f"Zu viele Rechenpunkte: maximal {MAX_POINTS:,}. dt vergrößern oder Dauer kürzen.")
    if c["kind"] in {"thermal","tank","gas","flow","motor"} and c["u_min"]<0:
        errors.append("Dieses Modell verwendet eine unidirektionale Ansteuerung von 0 bis 100 %.")
    if c["kind"] in {"PT1","PT2"} and c["ks"]==0:errors.append("Ks darf nicht null sein.")
    if c["kind"]=="motor":
        if c["motor_sync_rpm"]<=c["motor_rpm"]:errors.append("Synchrondrehzahl muss größer als die Nenndrehzahl sein.")
        if c["motor_start_ratio"]<=0 or c["motor_breakdown_ratio"]<1:errors.append("Motor-Momentkennlinie ist unplausibel.")
        data=motor_data(c) if not errors else None
        if data:
            maxspeed=c["motor_rpm"]*data["speed_scale"]
            if not 0<=c["setpoint"]<=maxspeed:errors.append("Drehzahl-/Bandsollwert liegt außerhalb des Nennbereichs.")
            if c["y0"]<0:errors.append("Anfangsdrehzahl darf nicht negativ sein.")
            if data["drop"]>=c["motor_voltage"]:errors.append("Berechneter Spannungsfall erreicht die Versorgungsspannung.")
            if data["drop"]>0.03*c["motor_voltage"]:warnings.append("Spannungsfall über 3 %: überschlägiger Hinweis, keine Kabelbemessung.")
    if c["kind"] in {"tank","mechanics"}:
        limit=c["tank_height"] if c["kind"]=="tank" else c["mech_hub"]
        if not 0<=c["setpoint"]<=limit:errors.append("Sollwert liegt außerhalb des physikalischen Hub-/Füllbereichs.")
        if not 0<=c["y0"]<=limit:errors.append("Anfangswert liegt außerhalb des physikalischen Hub-/Füllbereichs.")
    if c["kind"]=="gas":
        if not 0<=c["setpoint"]<=c["gas_pmax"]:errors.append("Solldruck liegt außerhalb des Betriebsdruckbereichs.")
        if not 0<=c["y0"]<=c["gas_pmax"]:errors.append("Anfangsdruck liegt außerhalb des Betriebsdruckbereichs.")
    if c["kind"]=="thermal":
        if min(c["y0"],c["setpoint"],c["temp_ambient"],c["temp_inlet"])<=-273.15:
            errors.append("Temperaturen müssen oberhalb des absoluten Nullpunkts liegen.")
    if errors:return errors,warnings
    m=physical_model(c)
    required=equilibrium_input(c,m,c["setpoint"])
    if not c["u_min"]-1e-8<=required<=c["u_max"]+1e-8:
        warnings.append(f"Sollwert stationär nicht erreichbar: benötigt ca. {required:.4g} %, verfügbar {c['u_min']:g}…{c['u_max']:g} %.")
    if c["kind"]=="thermal":
        sign=1 if c["temp_mode"]=="Heizen" else -1
        if sign*(c["setpoint"]-m["offset"])< -1e-9:
            errors.append("Thermische Betriebsart passt nicht zur Solltemperatur gegenüber dem passiven Gleichgewicht.")
    if c["kind"]=="flow" and (c["y0"]<0 or c["setpoint"]<0):errors.append("Durchfluss muss nichtnegativ sein.")
    if c["kind"]=="flow" and c["setpoint"]>flow_data(c,c["u_max"])["q"]:
        warnings.append("Solldurchfluss mit verfügbarer Druckdifferenz und Rohr-/Ventilwiderständen nicht erreichbar.")
    if c["kind"]=="motor":
        data=motor_data(c)
        available=data["torque"]
        if c["motor_drive"]=="Direktmotor: Kennlinienmodell":
            available*=c["motor_start_ratio"]*max(0,1-data["drop"]/c["motor_voltage"])**2
        if c["motor_load"]>=available: warnings.append("Lastmoment verhindert den Hochlauf; der Simulator zeigt Stillstand statt einer erfolgreichen Drehzahlkurve.")
    if c["actuator_mode"] in {"Ein/Aus","Stufen"}:
        warnings.append("Diskrete Ansteuerung kann bleibende Schwingungen verursachen. Überschwing- und Genauigkeitsanforderungen müssen gesondert erfüllt werden.")
    if c["strategy"]=="PI" and c["ki"]==0:
        warnings.append("Ki ist null: kein wirksamer I-Anteil; eine bleibende Abweichung kann auftreten.")
    if c["disturbance_position"]!="Keine Störung" and c["disturbance_time"]>=c["t_end"]:
        warnings.append("Störung liegt außerhalb des ausgewerteten Zeitfensters; keine Störantwort vorhanden.")
    poles=loop_poles(c,m)
    if len(poles) and max(poles.real)>=-1e-10:
        warnings.append("Linearisiertes geschlossenes Modell ist instabil oder grenzstabil. Eine positive Reglereignung wird nicht bestätigt.")
    return errors,warnings


def time_resolution(c,m):
    """Plant and controller rate checks; fast exact sensor/actuator filters need no Euler bound."""
    A,B=m["A"],m["B"]
    plant_rate=max(abs(np.linalg.eigvals(A)))
    if len(B)==1:
        control_rate=abs(A[0,0])+abs(c["kp"]*B[0])+np.sqrt(abs(c["ki"]*B[0]))
    else:
        control_rate=max(plant_rate,np.sqrt(abs(A[1,0])+abs(c["kp"]*B[1])),abs(c["kd"]*B[1]))
    max_step=.08/max(float(control_rate),1e-8)
    if c["kind"]=="motor":max_step=min(max_step,.01)
    if c["delay"]>0:max_step=min(max_step,c["delay"]/10)
    if c["controller"]=="PID" and c["kd"]!=0:max_step=min(max_step,c["derivative_filter"]/5)
    return max_step


def _transition(m,h,sensor_tau):
    """Exact ZOH propagation of linear plant + first-order sensor; affine input columns."""
    n=len(m["B"]);ns=n+(sensor_tau>0)
    A=np.zeros((ns,ns));A[:n,:n]=m["A"]
    if sensor_tau>0:A[n,:n]=m["C"]/sensor_tau;A[n,n]=-1/sensor_tau
    H=np.zeros((ns,4));H[:n,0]=m["B"];H[:n,1]=m["D"];H[:n,2]=m["f"]
    if sensor_tau>0:H[n,2]=m["offset"]/sensor_tau;H[n,3]=1/sensor_tau
    aug=np.zeros((ns+4,ns+4));aug[:ns,:ns]=A;aug[:ns,ns:]=H
    E=expm(aug*h)
    return E[:ns,:ns],E[:ns,ns:]


def disturbance_unit(c):
    if c["disturbance_position"]=="Am Ausgang":return c["unit"]
    if c["disturbance_position"]=="Vor der Strecke":return "%"
    if c["kind"] in {"PT1","PT2"}:return c["unit"]+("/s" if c["kind"]=="PT1" else "/s²")
    return c["disturbance_unit"]


def simulate(c):
    """Sampled controller, bounded work, refined control grid, exact linear propagation.

    Display dt is a requested maximum step; internal refinement is reported.
    Sensor noise is uniform ±amplitude on the display/control sample and reproducible.
    Dead time acts on actuator output, not on disturbance channels.
    """
    errors,notices=validate_config(c)
    if errors:raise ModelError("\n".join(errors))
    c=tune_controller(c)
    errors,notices=validate_config(c)
    if errors:raise ModelError("\n".join(errors))
    m=physical_model(c)
    npoints=int(math.ceil(c["t_end"]/c["dt"]))+1
    if npoints>MAX_POINTS:raise ModelError("Rechenpunktgrenze überschritten.")
    t=np.linspace(0,c["t_end"],npoints)
    display_h=t[1]-t[0]
    sub=max(1,int(math.ceil(display_h/time_resolution(c,m))))
    if sub*(npoints-1)>MAX_INTERNAL_STEPS:
        raise ModelError("Dynamik und Simulationsdauer erfordern zu viele interne Schritte. Dauer kürzen oder schnelle Modellparameter prüfen.")
    h=display_h/sub
    if sub>1:notices.append(f"Interne Reglerauflösung automatisch auf {h:.6g} s verfeinert (Ausgabe {display_h:.6g} s).")
    n=len(m["B"])
    state=np.r_[m["x0"],c["y0"]] if c["sensor_tau"]>0 else m["x0"].copy()
    E,H=_transition(m,h,c["sensor_tau"])
    # Flow has nonlinear static characteristic but exact first-order/sensor dynamics.
    if c["kind"]=="flow":
        fm=copy.deepcopy(m);fm["B"]=np.array([1/c["flow_tau"]]);E,H=_transition(fm,h,c["sensor_tau"])
    rng=np.random.default_rng(int(c["seed"]))
    history=deque()
    initial_u=float(np.clip(c["bias"],c["u_min"],c["u_max"]))
    actuator_state=initial_u
    integral=initial_u
    filtered=c["y0"];on=False
    y=c["y0"];measurement=float(np.clip(y,c["sensor_min"],c["sensor_max"]))
    records=[];boundary_hits=0;sensor_hits=0;pressure_hits=0
    md=motor_data(c) if c["kind"]=="motor" else None
    def event(time):
        return c["disturbance_value"] if c["disturbance_position"]!="Keine Störung" and time>=c["disturbance_time"] else 0.
    def reference(time):
        ramp=c["motor_ramp"] if c["kind"]=="motor" else c["setpoint_ramp"]
        return c["y0"]+(c["setpoint"]-c["y0"])*min(1.,time/ramp) if ramp>0 else c["setpoint"]
    def row(time,u,raw,applied,d):
        return [time,reference(time),y,measurement,u,raw,applied,reference(time)-measurement,d]
    records.append(row(0,initial_u,initial_u,initial_u,event(0)))
    raw=initial_u;applied=initial_u;d=0.
    direction=1 if (m["B"][int(np.argmax(abs(m["B"])))])>0 else -1
    for k in range(1,npoints):
        noise=rng.uniform(-c["noise"],c["noise"])
        for j in range(sub):
            now=t[k-1]+j*h
            # Disturbance onset is quantized forward to the internal grid (error < h).
            w=reference(now);err=w-measurement
            p=c["kp"]*(c["beta"]*(w-c["y0"])-(measurement-c["y0"]))
            filtered += (measurement-filtered)*(-math.expm1(-h/c["derivative_filter"]))
            dp=-c["kd"]*(measurement-filtered)/c["derivative_filter"] if c["controller"]=="PID" else 0.
            ki=c["ki"] if c["controller"] in {"PI","PID"} else 0.
            candidate=integral+ki*err*h
            raw=c["bias"]+p+dp if c["controller"]=="P" else p+candidate+dp
            if c["controller"]=="Zweipunkt":
                if direction*err>c["hysteresis"]/2:on=True
                elif direction*err< -c["hysteresis"]/2:on=False
                raw=c["u_max"] if on else c["u_min"]
            demand=float(np.clip(raw,c["u_min"],c["u_max"]))
            hard_limit=raw-demand
            if c["actuator_mode"]=="Ein/Aus":
                demand=c["u_max"] if demand>(c["u_min"]+c["u_max"])/2 else c["u_min"]
            elif c["actuator_mode"]=="Stufen":
                fraction=(demand-c["u_min"])/(c["u_max"]-c["u_min"])
                demand=c["u_min"]+round(fraction*c["stages"])/c["stages"]*(c["u_max"]-c["u_min"])
            step=demand-actuator_state
            if c["actuator_tau"]>0:step*=(-math.expm1(-h/c["actuator_tau"]))
            unconstrained_step=step
            step=float(np.clip(step,-c["rate"]*h,c["rate"]*h))
            rate_limit=unconstrained_step-step
            actuator_state=float(np.clip(actuator_state+step,c["u_min"],c["u_max"]))
            # Integrator is frozen for output/rate saturation only when it drives further into that limit.
            if ki and (hard_limit*(ki*err)>1e-10 or rate_limit*(ki*err)>1e-10):
                pass
            elif ki:integral=candidate
            history.append((now,actuator_state))
            target=now-c["delay"]
            while len(history)>1 and history[1][0]<=target+1e-12:history.popleft()
            applied=history[0][1] if history[0][0]<=target+1e-12 else initial_u
            d=event(now)
            process_d=d if c["disturbance_position"]=="Prozessstörung" else 0.
            output_d=d if c["disturbance_position"]=="Am Ausgang" else 0.
            if c["disturbance_position"]=="Vor der Strecke":applied+=d
            if c["kind"]=="motor":
                def acceleration(speed):
                    nrpm=max(0.,speed/md["speed_scale"])
                    torque=md["torque"]*applied/100
                    if c["motor_drive"]=="Direktmotor: Kennlinienmodell":
                        ns=c["motor_sync_rpm"]
                        # Entered characteristic points, no claim of a universal induction-motor curve.
                        kink=min(.8*ns,.95*c["motor_rpm"])
                        torque=np.interp(nrpm,[0,kink,c["motor_rpm"],ns],
                            [c["motor_start_ratio"]*md["torque"],c["motor_breakdown_ratio"]*md["torque"],md["torque"],0.])
                        torque*=max(0,1-md["drop"]/c["motor_voltage"])**2*(applied/100)
                    omega=nrpm*2*np.pi/60
                    net=torque-c["motor_load"]-process_d-c["motor_friction"]*omega
                    return net/c["motor_inertia"]*60/(2*np.pi)*md["speed_scale"]
                a=acceleration(state[0]);b=acceleration(state[0]+h*a/2)
                cc=acceleration(state[0]+h*b/2);dd=acceleration(state[0]+h*cc)
                new_speed=max(0.,state[0]+h*(a+2*b+2*cc+dd)/6)
                state[0]=new_speed
                if c["sensor_tau"]>0:state[n]+=(new_speed+output_d-state[n])*(-math.expm1(-h/c["sensor_tau"]))
            else:
                driver=flow_data(c,applied,process_d)["q"] if c["kind"]=="flow" else applied
                state=E@state+H@np.array([driver,process_d,1.,output_d])
            value=state[0]+m["offset"]
            if m["lower"] is not None and value<m["lower"]:
                state[0]=m["lower"]-m["offset"];boundary_hits+=1
                if n==2 and state[1]<0:state[1]=0.
            if m["upper"] is not None and value>m["upper"]:
                state[0]=m["upper"]-m["offset"];boundary_hits+=1
                if n==2 and state[1]>0:state[1]=0.
            y=float(state[0]+m["offset"]+output_d)
            if c["kind"]=="gas" and y>c["gas_pmax"]:pressure_hits+=1
            measured=float(state[n]) if c["sensor_tau"]>0 else y
            before_clip=measured+noise
            measurement=float(np.clip(before_clip,c["sensor_min"],c["sensor_max"]))
            sensor_hits+=int(measurement!=before_clip)
            if not np.isfinite(state).all() or not np.isfinite([raw,integral,measurement]).all() or abs(y)>1e12:
                raise ModelError("Simulation divergiert oder überschreitet den numerischen Bereich. Regler, Modell und Zeitauflösung prüfen.")
        # The disturbance column describes the physical interval just integrated.
        records.append(row(t[k],actuator_state,raw,applied,d))
    if boundary_hits:notices.append("Physikalische Grenze erreicht: Leerstand/Überlauf oder mechanischer Anschlag. Am Anschlag wird Bewegung begrenzt; keine positive Eignungsbestätigung.")
    if sensor_hits:notices.append("Messbereich wurde überschritten: Messsignal ist gesättigt.")
    if pressure_hits:notices.append("Zulässiger Betriebsdruck wurde überschritten; das Modell enthält kein Sicherheitsventil.")
    df=pd.DataFrame(records,columns=["Zeit [s]","Sollwert w","Regelgröße y","Messwert x","Stellgröße u",
        "Reglerausgang roh","Streckeneingang u","Regeldifferenz e","Störung d"])
    # Same-time error and exact endpoint are guaranteed in the exported data.
    df["Regeldifferenz e"]=df["Sollwert w"]-df["Messwert x"]
    df.attrs.update(config=c,notices=notices,internal_dt=h,unit=c["unit"],
                    output_disturbance_unit=disturbance_unit(c))
    return df


def calculate_metrics(df,c):
    y=df["Regelgröße y"].to_numpy();t=df["Zeit [s]"].to_numpy()
    target=c["setpoint"];amplitude=target-c["y0"]
    span=c["sensor_max"]-c["sensor_min"]
    tolerance=max(abs(amplitude),span*.01)*c["tolerance_percent"]/100
    over=max(0.,float(np.max(np.sign(amplitude)*(y-target)))/abs(amplitude)*100) if abs(amplitude)>1e-12 else 0.
    # Linear O(N) last violation; require a finite observation interval inside the band.
    last=np.flatnonzero(np.abs(y-target)>tolerance)
    index=int(last[-1]+1) if len(last) else 0
    min_hold=max((t[-1]-t[0])*.05,5*(t[1]-t[0]))
    settling=float(t[index]) if index<len(t) and t[-1]-t[index]>=min_hold else None
    tail=y[max(0,len(y)-max(10,len(y)//10)):]
    stationary=float(np.ptp(tail))<=tolerance*.25
    return dict(final=float(y[-1]),residual=float(target-y[-1]),overshoot=over,
                settling=settling,tolerance=tolerance,stationary=stationary,
                steady_error=float(target-y[-1]) if stationary else None,
                achieved=settling is not None and (not c["no_overshoot"] or over<=c["overshoot_limit"]))



# Explizite physikalische Größen: keine fehleranfällige Teilwortsuche.
# tuple: unit, initial/passive baseline, target, signed gain, time constant, sensor min/max.
QUANTITIES = {
    "Kanal-Differenzdruck": ("Pa",0.,250.,5.,3.,0.,600.),
    "CO₂ / Luftqualität": ("ppm",1400.,800.,-10.,300.,300.,2000.),
    "CO₂-geführte Lüftung": ("ppm",1400.,800.,-10.,300.,300.,2000.),
    "Raum- oder Zuluftfeuchte": ("% r. F.",30.,50.,.5,120.,0.,100.),
    "Raumfeuchte": ("% r. F.",30.,50.,.5,120.,0.,100.),
    "Überhitzungsregelung": ("K",15.,6.,-.12,10.,0.,30.),
    "Sauerstoff im Belebungsbecken": ("mg/l",.5,2.,.06,120.,0.,10.),
    "pH-Wert": ("pH",10.,7.,-.06,60.,0.,14.),
    "Leitfähigkeit": ("µS/cm",100.,500.,10.,60.,0.,2000.),
    "Konzentration": ("%",10.,50.,.8,60.,0.,100.),
    "Taupunkt": ("°C",10.,-20.,-.5,120.,-60.,30.),
    "Drehmoment": ("Nm",0.,50.,1.,.2,-10.,120.),
    "Gleichlauf / Synchronisation": ("1/min Differenz",50.,0.,-1.,1.,-100.,100.),
    "Mischungsverhältnis": ("Verhältnis",.5,1.,.015,5.,0.,3.),
    "Geschwindigkeit": ("mm/s",0.,50.,1.,.5,0.,150.),
    "Kraft": ("N",0.,1000.,20.,.5,0.,2500.),
    "Leistungsbegrenzung": ("kW",200.,100.,-2.,10.,0.,300.),
    "Eigenverbrauchsoptimierung": ("kW",50.,0.,-1.,10.,-100.,100.),
    "Speicherladung": ("% Ladezustand",20.,80.,1.,600.,0.,100.),
    "Lastmanagement": ("kW",400.,250.,-3.,10.,0.,500.),
    "Druckhaltung": ("bar Überdruck",1.,5.,.09,3.,0.,12.),
    "Pumpenkaskade": ("bar Überdruck",1.,5.,.09,3.,0.,12.),
    "Systemdruck": ("bar Überdruck",5.,80.,1.5,1.,0.,200.),
    "Verdampfungsdruck": ("bar Überdruck",5.,3.,-.04,3.,0.,10.),
    "Verflüssigungsdruck": ("bar Überdruck",20.,12.,-.12,8.,0.,30.),
    "Dampfdruck": ("bar Überdruck",2.,10.,.14,30.,0.,20.),
    "Dampftemperatur": ("°C",150.,180.,.5,30.,100.,250.),
    "Kesselwasserstand": ("m",.5,1.,.015,20.,0.,2.),
    "Kaskade Produkt/Medium": ("°C",20.,120.,2.,600.,0.,250.),
    "Zonentemperatur": ("°C",20.,180.,3.,300.,0.,400.),
    "Heiz-/Kühlsequenz": ("°C",28.,22.,-.1,600.,0.,40.),
    "Mischlufttemperatur": ("°C",5.,18.,.2,10.,-10.,35.),
    "Rücklauftemperaturbegrenzung": ("°C",60.,45.,-.3,60.,0.,100.),
    "witterungsgeführter Heizkreis": ("°C",20.,45.,.5,120.,0.,90.),
}

CONCEPT_VARIANTS = {
    ("Dampf","Dampfdruck"),("Dampf","Dampftemperatur"),("Dampf","Kesselwasserstand"),
    ("RLT / Lüftung","Mischlufttemperatur"),("Heizung","witterungsgeführter Heizkreis"),
    ("Heizung","Rücklauftemperaturbegrenzung"),("Raumautomation","Heiz-/Kühlsequenz"),
    ("Prozesswärme","Zonentemperatur"),("Prozesswärme","Kaskade Produkt/Medium"),
    ("Kälte","Verdampfungsdruck"),("Kälte","Verflüssigungsdruck"),
    ("Wasser","Druckhaltung"),("Wasser","Pumpenkaskade"),
    ("Hydraulik / Pneumatik","Systemdruck"),("Hydraulik / Pneumatik","Geschwindigkeit"),
    ("Dosierung / Chemie","Mischungsverhältnis"),("Elektroantrieb","Gleichlauf / Synchronisation"),
}


def apply_actuator(c,name):
    """Sichtbare Klassenannahmen; keine implizite Herstellerparametrierung."""
    d=copy.deepcopy(c);d["actuator"]=name
    d.update(actuator_mode="Stetig",actuator_tau=0.,rate=10000.)
    lower=name.casefold()
    discrete=any(token in lower for token in ["ohne fu","magnetventil","schütz","last-leerlauf","lastabwurf","freigaben"])
    staged=any(token in lower for token in ["stufen","mehrstufig","kaskade","grundlast-"])
    if discrete:d.update(actuator_mode="Ein/Aus",rate=10000.)
    elif staged:d.update(actuator_mode="Stufen",stages=3,rate=10000.)
    elif any(token in lower for token in ["ventil","klappe","mischer"]):d.update(rate=10.,actuator_tau=.2)
    elif "servo" in lower or "schrittmotor" in lower:d.update(rate=2000.,actuator_tau=.005)
    else:d.update(rate=100.,actuator_tau=.05)
    if d["kind"]=="motor":
        direct="ohne fu" in lower
        d["motor_drive"]="Direktmotor: Kennlinienmodell" if direct else "FU: Momentenbetrieb"
        d["actuator_mode"]="Ein/Aus" if direct else "Stetig"
        d["actuator_tau"]=0.;d["rate"]=10000.
    if d["concept"]:
        # Passive baseline and gain remain explicit. The actuator name does not certify a process model.
        if "pH" in d["unit"] and "lauge" in lower:d.update(baseline=4.,y0=4.,ks=.06)
        elif "pH" in d["unit"] and "säure" in lower:d.update(baseline=10.,y0=10.,ks=-.06)
    return d


def make_preset(category,variant,actuator=None,strategy="Automatische Empfehlung"):
    profile=PRACTICAL_PROCESS_CATALOG[category][variant]
    d=default_config()
    d.update(category=category,variant=variant,title=f"{category} · {variant}",
             u_min=0.,u_max=100.,strategy=strategy,sensor_tau=.2,
             disturbance_position="Keine Störung",disturbance_value=0.,
             concept=profile["model"]=="Generische Prozessstrecke" or (category,variant) in CONCEPT_VARIANTS)
    model=profile["model"]
    if d["concept"]:
        if variant not in QUANTITIES:raise ModelError(f"Keine geprüfte Größen-Vorgabe für {category}/{variant}.")
        unit,initial,target,gain,tau,lo,hi=QUANTITIES[variant]
        d.update(kind="PT1",unit=unit,baseline=initial,y0=initial,setpoint=target,
                 ks=gain,ts=tau,sensor_min=lo,sensor_max=hi,t_end=10*tau,
                 disturbance_unit=unit+"/s",
                 model_note="Konzeptvorlage mit ausdrücklich angenommenem PT1-Ersatzmodell. Keine physikalische Prognose der genannten Anlage; Ks, Arbeitspunkt und Ts müssen aus Messungen identifiziert werden.",
                 assumptions=["Die Zahlen sind Lehrbeispiele, keine Herstellerdaten.",
                              "Aktorname bezeichnet das Anwendungskonzept; die unten sichtbaren Dynamik-/Kennwerte bestimmen die Rechnung."])
        if variant=="Speicherladung":
            d["assumptions"].append("Dieses PT1-Lehrbeispiel ist ausdrücklich kein Batterie-Ladezustandsmodell; eine Energie-/Kapazitätsbilanz ist nicht implementiert.")
    elif model=="Temperaturregelung":
        room=(category in {"RLT / Lüftung","Raumautomation"} and variant!="Zulufttemperatur" and variant!="Frostschutz") or variant in {"Raumtemperatur","Kühlraumtemperatur"}
        cooling=category=="Kälte" or variant=="Raumtemperatur Kühlen"
        d.update(kind="thermal",unit="°C",sensor_min=-20.,sensor_max=120.,
                 temp_mode="Kühlen" if cooling else "Heizen",disturbance_unit="W zusätzliche Wärmelast",
                 model_note="Thermische Energie- und Durchflussbilanz mit einer vollständig durchmischten äquivalenten Wärmekapazität. Keine automatisch ermittelte Gebäudemasse, Schichtung oder Phasenänderung.")
        if room:
            d.update(temp_medium="Luft",temp_rho=1.204,temp_cp=1005.,temp_volume=300.,
                temp_extra_capacity=20_000_000.,temp_loss=300.,temp_power=12.,temp_efficiency=1.,
                temp_ambient=32. if cooling else 5.,temp_inlet=32. if cooling else 5.,
                y0=28. if cooling else 18.,setpoint=24. if cooling else 21.)
            if variant=="Kühlraumtemperatur":d.update(temp_ambient=25.,temp_inlet=25.,y0=12.,setpoint=4.,temp_power=15.)
            d["assumptions"]=["Zusatzkapazität 20 MJ/K steht beispielhaft für Gebäude, Einrichtung oder Ware und muss angepasst werden."]
        elif category=="RLT / Lüftung":
            d.update(temp_medium="Luft",temp_rho=1.204,temp_cp=1005.,temp_volume=3.,
                     temp_extra_capacity=100_000.,temp_flow=5000.,temp_loss=50.,temp_power=40.,
                     temp_efficiency=1.,temp_ambient=5.,temp_inlet=5.,y0=10.,setpoint=21.)
            if variant=="Frostschutz":
                d["setpoint"]=8.;d["y0"]=5.
                d["assumptions"]=["Nur thermischer Regelversuch. Ein unabhängiger Frostschutz mit Abschaltungen ist nicht modelliert."]
        elif category=="Prozesswärme":
            d.update(temp_medium="Benutzerdefiniert",temp_rho=7800.,temp_cp=500.,
                     temp_volume=.2,temp_extra_capacity=0.,temp_power=150.,temp_loss=500.,
                     temp_ambient=20.,temp_inlet=20.,y0=20.,setpoint=180.,sensor_max=400.)
            d["assumptions"]=["Äquivalente feste Wärmekapazität; Wärmeverlustkoeffizient gilt lokal. Keine separate Strahlungs-/Produktgeometrie."]
        else:
            d.update(temp_medium="Wasser",temp_volume=.5,temp_power=30.,temp_ambient=20.,temp_inlet=20.,
                     y0=20.,setpoint=55.,temp_flow=0.)
            if cooling:d.update(temp_ambient=12.,temp_inlet=12.,y0=12.,setpoint=6.,temp_power=12.)
            d["assumptions"]=["Durchmischter Wasserspeicher; Volumenstrom, Eintrittstemperatur und Zusatzmasse sind separat einzugeben."]
        d["t_end"]=10*(d["temp_rho"]*d["temp_volume"]*d["temp_cp"]+d["temp_extra_capacity"])/(d["temp_loss"]+d["temp_flow"]*d["temp_rho"]*d["temp_cp"]/3600)
    elif model=="Füllstandsregelung":
        pumping=variant=="Pumpensumpf-Füllstand"
        d.update(kind="tank",unit="m",sensor_min=0.,sensor_max=2.5,
                 tank_mode="Ablauf regeln" if pumping else "Zulauf regeln",
                 y0=2. if pumping else .3,setpoint=1.5,t_end=7200.,
                 disturbance_unit="m³/h zusätzlicher Zulauf" if pumping else "m³/h zusätzlicher Abfluss",
                 model_note="Mengenbilanz eines Behälters mit konstantem Querschnitt. Leerstand und Überlauf werden als Grenzereignisse ausgewiesen; keine fiktive PT1-Gleichgewichtshöhe.")
    elif model=="Drehzahlregelung":
        band=variant=="Bandgeschwindigkeit"
        d.update(kind="motor",unit="m/s" if band else "1/min",motor_band=band,
                 motor_gear=10. if band else 1.,motor_diameter=.2,
                 y0=0.,setpoint=1. if band else 1200.,sensor_min=0.,sensor_max=2. if band else 1800.,
                 t_end=40.,disturbance_unit="Nm zusätzliches Lastmoment",
                 model_note="Mechanische Momentenbilanz. FU/EC/Servo wird als idealer geregelter Momentensteller bis zum Nennbereich angenähert. Direktmotor verwendet die ausdrücklich einstellbaren Momentkennlinienpunkte; kein allgemeines Asynchronmotor-Ersatzschaltbild.")
    elif model=="Druckregelung" and category=="Druckluft":
        d.update(kind="gas",unit="bar Überdruck",y0=0.,setpoint=7.,sensor_min=0.,sensor_max=12.,
                 t_end=2400.,disturbance_unit="m³/h zusätzlicher Referenzverbrauch",
                 model_note="Isotherme ideale Gas-Massenbilanz. Förder- und Verbrauchsvolumenstrom gelten bei den eingegebenen Referenzbedingungen; alle Drücke sind in der Anzeige Überdrücke. Kein Sicherheitsventil oder Kondensationsmodell.")
    elif model=="Durchflussregelung":
        air=category in {"RLT / Lüftung","Raumautomation"}
        dosing=variant in {"Dosiermenge","Chemikaliendosierung"}
        d.update(kind="flow",unit="l/h" if dosing else "m³/h",y0=0.,setpoint=50. if dosing else 60.,
                 sensor_min=0.,sensor_max=200.,t_end=60.,disturbance_unit="bar zusätzliche verfügbare Druckdifferenz",
                 model_note="Quadratische Rohr-/Ventilkennlinie bei konstanter Druckquelle plus separat identifizierte Durchflusszeitkonstante. Rohr-Aufenthaltszeit ist ausschließlich eine Zusatzinformation.")
        if air:d.update(flow_qref=10000.,flow_diameter=500.,flow_length=30.,flow_density=1.204,
                        flow_supply_dp=.015,flow_valve_dp=.004,setpoint=6000.,sensor_max=20000.)
        if dosing:d.update(flow_qref=.1,flow_diameter=10.,flow_length=5.,flow_scale=1000.,flow_valve_dp=2.,flow_supply_dp=4.)
    elif model=="Position / Mechanik":
        d.update(kind="mechanics",unit="mm",y0=0.,setpoint=250.,sensor_min=0.,sensor_max=600.,
                 u_min=-100.,t_end=120.,disturbance_unit="N zusätzliche Gegenkraft",
                 model_note="Kraftangeregtes Masse-Feder-Dämpfer-System mit beidseitigem Kraftsteller und inelastischen Hubanschlägen. Ein Motor/Getriebe oder Hydraulikventil wird durch die verfügbare Lastkraft beschrieben; innere Servoregelungen sind nicht enthalten.")
    else:raise ModelError(f"Für {category}/{variant} ist kein eindeutiges Modell definiert.")
    d["dt"]=max(1e-6,d["t_end"]/4000.)
    d["disturbance_time"]=d["t_end"]/2
    d=apply_actuator(d,actuator or profile["actuators"][0])
    if strategy=="Stufen-/Kaskadensteuerung":d["actuator_mode"]="Stufen"
    return tune_controller(d)


# Gemeinsame Eingabemetadaten für alle Ansichten; validiert wird zusätzlich zentral.
# label, minimum (None permits a signed value), help text.
FIELDS = {
    "thermal": [
        ("temp_volume","Mediumvolumen [m³]",1e-6,"Volumen des vollständig durchmischten Mediums."),
        ("temp_power","Thermische Nennleistung [kW]",1e-6,"Thermische Nutz-/Quellenleistung vor η; keine elektrische Verdichteraufnahme."),
        ("temp_ambient","Umgebungstemperatur [°C]",-273.14,"Temperatur der Umgebung für den Wärmeübergang."),
        ("temp_loss","Wärmeübergang an Umgebung [W/K]",1e-6,"UA als konzentrierter Verlust-/Gewinnkoeffizient."),
        ("temp_efficiency","Thermischer Wirkungsgrad [0–1]",0.,"0 < η ≤ 1. Bei gegebener Nutzleistung η=1; kein COP."),
        ("temp_extra_capacity","Zusätzliche Wärmekapazität [J/K]",0.,"Gebäude, Behälterwand oder Produkt; zusätzlich zum Medium."),
        ("temp_flow","Durchströmung [m³/h]",0.,"Gleicher Zu- und Abfluss; Medienvolumen bleibt konstant."),
        ("temp_inlet","Eintrittstemperatur [°C]",-273.14,"Temperatur des einströmenden Mediums."),
        ("temp_rho","Dichte [kg/m³]",1e-6,"Stoffwert für das betrachtete Temperaturniveau."),
        ("temp_cp","Spezifische Wärmekapazität [J/(kg·K)]",1e-6,"Achtung: J, nicht kJ. Wasser etwa 4180."),
    ],
    "tank": [
        ("tank_volume","Behältervolumen [m³]",1e-6,"Volumen bis zur maximalen Füllhöhe."),
        ("tank_height","Maximale Füllhöhe [m]",1e-6,"Konstanter Querschnitt A=V/H."),
        ("tank_qmax","Maximaler geregelter Volumenstrom [m³/h]",1e-6,"Bei 100 %; linear zur stetigen Ansteuerung."),
        ("tank_baseflow","Ungeregelter Gegenstrom [m³/h]",0.,"Abfluss bei Zulaufregelung; Zufluss bei Ablaufregelung."),
    ],
    "motor": [
        ("motor_power","Mechanische Motornennleistung [kW]",1e-6,"Abgabeleistung an der Welle, nicht elektrische Aufnahme."),
        ("motor_rpm","Nenndrehzahl [1/min]",1e-6,"Nenndaten bestimmen M=P/ω."),
        ("motor_sync_rpm","Synchrondrehzahl Direktmodell [1/min]",1e-6,"Nur bei Direktkennlinie dynamisch relevant; muss größer als nN sein."),
        ("motor_inertia","Gesamtträgheitsmoment auf Motorwelle [kg·m²]",1e-8,"Motor und auf die Motorwelle umgerechnete Last."),
        ("motor_load","Konstantes Lastmoment [Nm]",0.,"Entgegen der positiven Drehrichtung."),
        ("motor_friction","Viskose Reibung [Nm/(rad/s)]",0.,"Reibmoment b·ω."),
        ("motor_ramp","Sollwert-Rampenzeit [s]",0.,"Rampenzeit vom Anfangs- zum Sollwert. 0 = Sprung."),
        ("motor_eta","Motorwirkungsgrad η [0–1]",0.,"Nur Nennstrom-/Leitungsrechnung; 0 < η ≤ 1."),
        ("motor_cosphi","Leistungsfaktor cos φ [0–1]",0.,"Sinusförmiger stationärer Nennbetrieb; 0 < cos φ ≤ 1."),
        ("motor_voltage","Außenleiterspannung [V]",1e-6,"Symmetrisches Drehstromsystem."),
        ("motor_length","Einfache Cu-Leitungslänge [m]",0.,"Spannungsfall inklusive cos φ und X sin φ; bei FU nur Information."),
        ("motor_section","Cu-Leiterquerschnitt [mm²]",1e-6,"Keine Strombelastbarkeits-/Schutzorganberechnung."),
        ("motor_cable_temp","Leitertemperatur [°C]",-100.,"Temperaturkorrektur des Widerstands."),
        ("motor_reactance","Leitungsreaktanz [Ω/km]",0.,"Herstellerangabe; 0,08 ist nur eine typische Annahme."),
        ("motor_start_ratio","Anlaufmoment / Nennmoment",1e-6,"Nur Direktmodell. Kennlinienwert muss zum Motor passen."),
        ("motor_breakdown_ratio","Kippmoment / Nennmoment",1.,"Nur Direktmodell; Stützpunkt bei ca. 80 % Synchrondrehzahl."),
        ("motor_gear","Getriebeübersetzung Motor/Abtrieb",1e-6,"Nur für Bandgeschwindigkeit."),
        ("motor_diameter","Antriebsrollendurchmesser [m]",1e-6,"Nur für Bandgeschwindigkeit."),
    ],
    "gas": [
        ("gas_volume","Behältervolumen [m³]",1e-6,"Konstantes geometrisches Gasvolumen."),
        ("gas_qmax","Förderstrom bei Referenzbedingungen [m³/h]",1e-6,"Gleiche Bezugsbedingungen wie Verbrauch."),
        ("gas_demand","Grundverbrauch bei Referenzbedingungen [m³/h]",0.,"Zusatzverbrauch über Prozessstörung."),
        ("gas_pmax","Maximaler Betriebsüberdruck [bar]",1e-6,"Prüfgrenze; kein fiktives Druckbegrenzungsventil."),
        ("gas_pref","Referenzdruck Volumenstrom [bar absolut]",1e-6,"Beispielsweise 1,01325 bar absolut; keine unbenannte Normierung."),
        ("gas_tref","Referenztemperatur Volumenstrom [K]",1e-6,"273,15 K=0 °C; 293,15 K=20 °C."),
        ("gas_temperature","Isotherme Behältertemperatur [K]",1e-6,"Für die gesamte Simulation konstant angenommen."),
        ("gas_ambient","Umgebungsdruck [bar absolut]",1e-6,"Überdruckanzeige bezieht sich auf diesen Wert."),
    ],
    "flow": [
        ("flow_qref","Referenzdurchfluss [m³/h]",1e-6,"Für die Druckverlustangaben. Anzeige kann in l/h umgerechnet sein."),
        ("flow_length","Rohrlänge [m]",0.,"Geht in den Druckverlust ein; nicht als PT1-Zeitkonstante verwendet."),
        ("flow_diameter","Rohrinnendurchmesser [mm]",1e-6,"Nicht gleichbedeutend mit DN."),
        ("flow_friction","Darcy-Reibungsbeiwert λ",0.,"Am Arbeitspunkt angenommener konstanter Reibungsbeiwert."),
        ("flow_density","Dichte [kg/m³]",1e-6,"Konstante Dichte; für Luft nur kleine relative Druckänderungen."),
        ("flow_valve_dp","Ventilverlust bei Qref und 100 % [bar]",1e-6,"Ventilleitwert linear zur Öffnung; Δp quadratisch mit Q."),
        ("flow_supply_dp","Verfügbare Druckdifferenz [bar]",1e-6,"Konstante Druckquelle im Modell; keine versteckte Pumpenkennlinie."),
        ("flow_tau","Identifizierte Durchfluss-Zeitkonstante [s]",1e-6,"Aus Messung/Herstellerdaten; keine Ventil-Vollhubzeit."),
    ],
    "mechanics": [
        ("mech_mass","Bewegte Masse [kg]",1e-8,"Wirksame translatorische Masse."),
        ("mech_spring","Federsteifigkeit [N/m]",1e-8,"Dieses Modell enthält eine Feder. Für freie Mechanik anderes Modell nötig."),
        ("mech_damping","Dämpfung [Ns/m]",0.,"Null ist zulässig und wird nicht künstlich angehoben."),
        ("mech_force","Kraft bei 100 % [N]",1e-8,"Direkte Kraftskalierung F=Fmax·u/100."),
        ("mech_load","Konstante Gegenkraft [N]",None,"Positiv wirkt gegen die positive Bewegungsrichtung."),
        ("mech_hub","Maximalhub [mm]",1e-8,"Inelastische Anschläge bei 0 und Maximalhub."),
        ("mech_v0","Anfangsgeschwindigkeit [mm/s]",None,"Zusätzlicher mechanischer Anfangszustand."),
    ],
    "PT1": [
        ("ks","Ks [Ausgabeeinheit/%]",None,"Vorzeichen bestimmt die Wirkrichtung; darf nicht null sein."),
        ("ts","PT1-Zeitkonstante [s]",1e-8,"Interne Rechnung behält volle Präzision."),
        ("baseline","Ausgang bei u=0 [Ausgabeeinheit]",None,"Absoluter Bezug/Arbeitspunkt; kein implizites Starten bei null."),
    ],
    "PT2": [
        ("ks","Ks [Ausgabeeinheit/%]",None,"Stationäre Verstärkung mit Vorzeichen."),
        ("zeta","Dämpfungsgrad ζ",0.,"Null ist möglich; Stabilitätsprüfung beachten."),
        ("omega0","Eigenkreisfrequenz ω0 [rad/s]",1e-8,"Keine abweichenden Widgetgrenzen zwischen den Arbeitsbereichen."),
        ("baseline","Ausgang bei u=0 [Ausgabeeinheit]",None,"Absoluter Bezug/Arbeitspunkt."),
    ],
}



# ------------------------ Benutzeroberfläche ------------------------
KIND_LABELS = {"PT1":"PT1-Lehrstrecke", "PT2":"PT2-Lehrstrecke", "thermal":"Thermische Bilanz",
               "tank":"Behälterbilanz", "gas":"Gasdruckbilanz", "flow":"Durchflussmodell",
               "motor":"Antrieb", "mechanics":"Masse–Feder–Dämpfer"}
OWNERS = ("simulation", "physical", "builder")


def init_session():
    if st.session_state.get("lab_version") != APP_VERSION:
        st.session_state.lab_version = APP_VERSION
        st.session_state.view = "Start"
        for owner in OWNERS:
            st.session_state[owner+"_config"] = default_config()
            st.session_state[owner+"_revision"] = 0
        category = next(iter(PRACTICAL_PROCESS_CATALOG))
        variant = next(iter(PRACTICAL_PROCESS_CATALOG[category]))
        st.session_state.physical_config = make_preset(category, variant)
        st.session_state.builder_graph = StreamlitFlowState([], [])


def config_of(owner):
    return st.session_state[owner+"_config"]


def widget_key(owner, field):
    return f"lab_{owner}_{st.session_state[owner+'_revision']}_{field}"


def replace_config(owner, config):
    st.session_state[owner+"_config"] = copy.deepcopy(config)
    st.session_state[owner+"_revision"] += 1
    st.session_state.pop(owner+"_result", None)


def save_field(owner, field, key):
    config_of(owner)[field] = st.session_state[key]


def number(owner, field, label, minimum=None, help=None, integer=False):
    c=config_of(owner);key=widget_key(owner,field)
    # Shared widgets never clamp imported physical values or change units.
    # Central validation gives an explicit error for invalid values.
    if integer:
        return st.number_input(label, value=int(c[field]), step=1, key=key,
                               on_change=save_field, args=(owner,field,key), help=help)
    return st.number_input(label, value=float(c[field]), format="%.8g", key=key,
                           on_change=save_field, args=(owner,field,key), help=help)


def choice(owner, field, label, options):
    c=config_of(owner);key=widget_key(owner,field)
    options=list(options)
    if c[field] not in options:options.insert(0,c[field])
    return st.selectbox(label, options, index=options.index(c[field]), key=key,
                        on_change=save_field,args=(owner,field,key))


def check(owner, field, label):
    key=widget_key(owner,field)
    return st.checkbox(label,value=bool(config_of(owner)[field]),key=key,
                       on_change=save_field,args=(owner,field,key))


def change_category(key):
    category=st.session_state[key]
    replace_config("physical",make_preset(category,next(iter(PRACTICAL_PROCESS_CATALOG[category]))))


def change_variant(key):
    c=config_of("physical")
    replace_config("physical",make_preset(c["category"],st.session_state[key]))


def change_actuator(owner,key):
    replace_config(owner,apply_actuator(config_of(owner),st.session_state[key]))


def set_medium(owner,key):
    c=config_of(owner);medium=st.session_state[key]
    c["temp_medium"]=medium
    if medium in {"Wasser","Luft"}:
        c["temp_rho"],c["temp_cp"]=(998.,4180.) if medium=="Wasser" else (1.204,1005.)
    replace_config(owner,c)


def load_free(owner):
    replace_config(owner,default_config())


def transfer_config(source,target):
    replace_config(target,config_of(source))
    if target=="builder":st.session_state.builder_graph=standard_graph(config_of(target))
    st.session_state.view={"simulation":"Simulation","physical":"Praxisvorgaben","builder":"Regelkreis bauen"}[target]


def configuration_controls(owner):
    c=config_of(owner)
    st.subheader("Modell und Vorgaben")
    if owner=="physical":
        key=widget_key(owner,"category")
        st.selectbox("Anwendungsgebiet",list(PRACTICAL_PROCESS_CATALOG),
                     index=list(PRACTICAL_PROCESS_CATALOG).index(c["category"]),key=key,
                     on_change=change_category,args=(key,))
        key=widget_key(owner,"variant")
        variants=list(PRACTICAL_PROCESS_CATALOG[c["category"]])
        st.selectbox("Regelungsaufgabe",variants,index=variants.index(c["variant"]),key=key,
                     on_change=change_variant,args=(key,))
    st.caption(c["title"]+" · "+KIND_LABELS[c["kind"]])
    if c["category"]:
        actuators=PRACTICAL_PROCESS_CATALOG[c["category"]][c["variant"]]["actuators"]
        key=widget_key(owner,"actuator")
        options=list(dict.fromkeys([c["actuator"]]+list(actuators)))
        st.selectbox("Stellglied / Antrieb",options,index=0,key=key,
                     on_change=change_actuator,args=(owner,key))
    if c["kind"] in {"PT1","PT2"}:
        choice(owner,"kind","Lehrstreckentyp",["PT1","PT2"])
        key=widget_key(owner,"unit")
        st.text_input("Ausgabeeinheit",value=c["unit"],key=key,
                      on_change=save_field,args=(owner,"unit",key))
    with st.expander("Streckenparameter",expanded=owner=="physical"):
        if c["kind"]=="thermal":
            key=widget_key(owner,"temp_medium")
            media=["Wasser","Luft","Benutzerdefiniert"]
            st.selectbox("Medium / Stoffwerte setzen",media,index=media.index(c["temp_medium"]),key=key,
                         on_change=set_medium,args=(owner,key))
            choice(owner,"temp_mode","Betriebsart",["Heizen","Kühlen"])
        if c["kind"]=="tank":choice(owner,"tank_mode","Geregelter Strom",["Zulauf regeln","Ablauf regeln"])
        if c["kind"]=="motor":
            choice(owner,"motor_drive","Motormodell",["FU: Momentenbetrieb","Direktmotor: Kennlinienmodell"])
            st.caption("FU-Modell: Momentenvorgabe, kein zusätzlich simuliertes FU-Drehzahlregelgerät. Direktmodell: eingegebene Momentkennlinie.")
        for field,label,minimum,helptext in FIELDS[c["kind"]]:
            number(owner,field,label,minimum,helptext)
    with st.expander("Sollwert, Anfangswert und Anforderungen",expanded=True):
        number(owner,"y0",f"Anfangswert [{c['unit']}]")
        number(owner,"setpoint",f"Sollwert [{c['unit']}]")
        if c["kind"]!="motor":number(owner,"setpoint_ramp","Sollwertrampe [s]")
        check(owner,"no_overshoot","Überschwingen begrenzen")
        number(owner,"overshoot_limit","Maximales Überschwingen [% des Sollsprungs]")
        number(owner,"tolerance_percent","Sollwert-Toleranz [% des Sollsprungs]")
    with st.expander("Regler",expanded=True):
        strategies=["Automatische Empfehlung","P","PI","PID","Zweipunkt","Stufen-/Kaskadensteuerung"]
        if c["category"]:
            strategies=list(dict.fromkeys(strategies+PRACTICAL_PROCESS_CATALOG[c["category"]][c["variant"]]["strategies"]))
        choice(owner,"strategy","Regelstrategie",strategies)
        check(owner,"automatic","Regler-Startwerte automatisch berechnen")
        if not c["automatic"]:
            choice(owner,"controller","Tatsächlich verwendeter Regler",["P","PI","PID","Zweipunkt"])
            for field,label in [("kp","Kp [%/Einheit]"),("ki","Ki [%/(Einheit·s)]"),
                                ("kd","Kd [%·s/Einheit]"),("beta","Sollwertgewicht β [0–1]"),
                                ("bias","Vorsteuerung / Anfangsstellwert [%]")]:number(owner,field,label)
        number(owner,"derivative_filter","D-Filter-Zeitkonstante [s]")
        number(owner,"hysteresis",f"Zweipunkt-Hysterese [{c['unit']}]")
        st.caption("Automatik liefert Startwerte. Erst die berechnete Kurve zeigt, ob die Anforderungen im untersuchten Zeitfenster erfüllt sind.")
    with st.expander("Stellglied, Totzeit und Messung"):
        choice(owner,"actuator_mode","Ansteuerung",["Stetig","Ein/Aus","Stufen"])
        if c["actuator_mode"]=="Stufen":number(owner,"stages","Zahl gleich großer Leistungsstufen",integer=True)
        for field,label in [("u_min","Stellgröße min. [%]"),("u_max","Stellgröße max. [%]"),
            ("rate","Maximale Stellgeschwindigkeit [%/s]"),("actuator_tau","Stellglied-Zeitkonstante [s]"),
            ("delay","Streckentotzeit [s]"),("sensor_tau","Messglied-Zeitkonstante [s]"),
            ("sensor_min",f"Messbereich min. [{c['unit']}]"),("sensor_max",f"Messbereich max. [{c['unit']}]"),
            ("noise",f"Messrauschen ± [{c['unit']}]")]:number(owner,field,label)
        number(owner,"seed","Zufallsstartwert",integer=True)
        st.caption("Rauschen ist gleichverteilt und reproduzierbar. Stufen bilden identische Leistungsanteile ab; keine Führungs-/Folgereglerkaskade.")
    with st.expander("Störung und Zeitfenster"):
        choice(owner,"disturbance_position","Störungsort",["Keine Störung","Vor der Strecke","Prozessstörung","Am Ausgang"])
        process_unit=c["unit"]+("/s²" if c["kind"]=="PT2" else "/s") if c["kind"] in {"PT1","PT2"} else c["disturbance_unit"]
        unit="%" if c["disturbance_position"]=="Vor der Strecke" else c["unit"] if c["disturbance_position"]=="Am Ausgang" else process_unit
        number(owner,"disturbance_value",f"Störsprung [{unit}]")
        number(owner,"disturbance_time","Störzeitpunkt [s]")
        number(owner,"t_end","Simulationsdauer [s]")
        number(owner,"dt","Maximaler Ausgabe-Zeitschritt [s]")
        st.caption(f"Maximal {MAX_POINTS:,} Ausgabepunkte und {MAX_INTERNAL_STEPS:,} interne Schritte. Grenzen gelten in allen Ansichten.")
    if owner!="physical":st.button("Neue freie Strecke",key=owner+"_free",on_click=load_free,args=(owner,))


def inspect_config(c):
    try:
        errors,warnings=validate_config(c)
        if errors:return c,errors,warnings,None
        effective=tune_controller(c)
        errors,warnings=validate_config(effective)
        return effective,errors,warnings,physical_model(effective)
    except (ValueError,ZeroDivisionError,OverflowError,np.linalg.LinAlgError) as exc:
        return c,["Modellparameter sind numerisch nicht auswertbar: "+str(exc)],[],None


def show_diagnostics(c):
    effective,errors,warnings,m=inspect_config(c)
    if c["concept"]:st.warning("Konzeptvorlage: Das PT1-Ersatzmodell ist nicht als physikalisches Modell dieser Anlage validiert.")
    st.info(c["model_note"])
    for assumption in c["assumptions"]:st.caption("Annahme: "+assumption)
    for err in errors:st.error(err)
    for warning in warnings:st.warning(warning)
    if m:
        with st.expander("Abgeleitete Modellwerte und Regler-Startwerte"):
            st.write({"Regler":effective["controller"],"Kp":effective["kp"],"Ki":effective["ki"],"Kd":effective["kd"],
                      "Sollwertgewicht β":effective["beta"],"Vorsteuerung [%]":effective["bias"],**m["metrics"]})
            st.caption("Lineare Stabilitätsprüfung umfasst Messglied und Stellglied-PT1. Totzeit, Begrenzungen, Stellrate und nichtlineare Kennlinien müssen über die Simulation bewertet werden.")
    return errors


def show_result(owner,c):
    errors=show_diagnostics(c)
    if st.button("Simulation berechnen",type="primary",key=owner+"_run",disabled=bool(errors)):
        try:
            with st.spinner("Regelkreis wird berechnet …"):
                data=simulate(c)
            st.session_state[owner+"_result"]=(fingerprint(c),data)
        except (ValueError,OverflowError,ZeroDivisionError,np.linalg.LinAlgError) as exc:
            st.error(str(exc));st.session_state.pop(owner+"_result",None)
    stored=st.session_state.get(owner+"_result")
    if not stored:return
    if stored[0]!=fingerprint(c):
        st.warning("Parameter wurden geändert. Bitte neu berechnen; die vorherige Kurve gehört zu anderen Einstellungen.")
        return
    df=stored[1];effective=df.attrs["config"];metrics=calculate_metrics(df,effective)
    for notice in df.attrs["notices"]:st.warning(notice)
    cols=st.columns(4)
    cols[0].metric("Endwert",f"{metrics['final']:.6g} {c['unit']}")
    cols[1].metric("Restabweichung am Ende",f"{metrics['residual']:.6g} {c['unit']}")
    cols[2].metric("Überschwingen",f"{metrics['overshoot']:.3g} %")
    cols[3].metric("Einschwingzeit",f"{metrics['settling']:.6g} s" if metrics["settling"] is not None else "nicht nachgewiesen")
    if metrics["stationary"]:st.caption(f"Im Schlussabschnitt annähernd stationär; beobachtete Abweichung {metrics['steady_error']:.6g} {c['unit']}.")
    else:st.caption("Kein ausreichend stationärer Schlussabschnitt: Eine bleibende Regelabweichung ist damit nicht nachgewiesen.")
    critical=any(any(word in notice for word in ["Grenze erreicht","Messbereich wurde","Betriebsdruck","instabil","nicht erreichbar","Hochlauf"]) for notice in df.attrs["notices"])
    if metrics["achieved"] and not critical:
        st.info("Sollwert-Toleranz und Überschwinggrenze sind in dieser Kurve erfüllt. Dies ist keine allgemeine Regler- oder Anlagenfreigabe.")
    else:st.warning("Die Anforderungen sind in diesem Zeitfenster nicht vollständig erfüllt.")
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout="constrained")
    for key in ["Sollwert w","Regelgröße y","Messwert x"]:axes[0].plot(df["Zeit [s]"],df[key],label=key,linewidth=1.4)
    axes[0].set_ylabel(c["unit"]);axes[0].legend();axes[0].grid(alpha=.25)
    for key in ["Stellgröße u","Streckeneingang u"]:axes[1].plot(df["Zeit [s]"],df[key],label=key)
    axes[1].set_ylabel("Stellgröße [%]");axes[1].set_xlabel("Zeit [s]");axes[1].legend();axes[1].grid(alpha=.25)
    st.pyplot(fig);plt.close(fig)
    st.caption(f"Interner Reglerzeitschritt ≤ {df.attrs['internal_dt']:.6g} s. Rauschen wird je Ausgabeintervall gehalten. Totzeiten werden auf dem internen Raster mit höchstens einem Schritt Verzögerungsfehler umgesetzt.")
    with st.expander("Messwerte und Export"):
        st.dataframe(df,width="stretch",hide_index=True)
        export=df.rename(columns={k:f"{k} [{c['unit']}]" for k in ["Sollwert w","Regelgröße y","Messwert x","Regeldifferenz e"]})
        export=export.rename(columns={**{k:f"{k} [%]" for k in ["Stellgröße u","Reglerausgang roh","Streckeneingang u"]},"Störung d":f"Störung d [{disturbance_unit(c)}]"})
        st.download_button("Messwerte als CSV",export.to_csv(index=False,sep=";",decimal=","),file_name="regelkreis_messwerte.csv",mime="text/csv",key=owner+"_csv")
        st.download_button("Einstellungen als JSON",json.dumps(effective,ensure_ascii=False,indent=2),file_name="regelkreis_einstellungen.json",mime="application/json",key=owner+"_json")


def graph_spec(c):
    labels={"soll":"Sollwert w", "summe":"Vergleich w − x", "regler":"Regler", "aktor":"Stellglied",
            "strecke":KIND_LABELS[c["kind"]], "ausgang":"Regelgröße y", "sensor":"Messglied x"}
    pairs={("soll","summe"),("summe","regler"),("regler","aktor"),("aktor","strecke"),
           ("strecke","ausgang"),("ausgang","sensor"),("sensor","summe")}
    if c["disturbance_position"]!="Keine Störung":
        labels["stoerung"]="Störung d"
        if c["disturbance_position"]=="Prozessstörung":pairs.add(("stoerung","strecke"))
        else:
            labels["stoersumme"]="Störsumme"
            before,after=("aktor","strecke") if c["disturbance_position"]=="Vor der Strecke" else ("strecke","ausgang")
            pairs.remove((before,after));pairs|={(before,"stoersumme"),("stoersumme",after),("stoerung","stoersumme")}
    return labels,pairs


def standard_graph(c):
    labels,pairs=graph_spec(c)
    positions={"soll":(0,0),"summe":(180,0),"regler":(360,0),"aktor":(540,0),
               "strecke":(900,0),"ausgang":(1250,0),"sensor":(540,200),
               "stoersumme":(720,0) if c["disturbance_position"]=="Vor der Strecke" else (1080,0),"stoerung":(720,-150)}
    nodes=[StreamlitFlowNode(id=id,pos=positions[id],data={"content":label},
                            node_type="input" if id in {"soll","stoerung"} else "default",
                            source_position="right",target_position="left") for id,label in labels.items()]
    edges=[StreamlitFlowEdge(id=f"{a}__{b}",source=a,target=b,animated=False) for a,b in sorted(pairs)]
    return StreamlitFlowState(nodes,edges)


def validate_graph(c,state):
    labels,expected=graph_spec(c)
    ids=[n.id for n in state.nodes];pairs=[(e.source,e.target) for e in state.edges]
    errors=[]
    if len(ids)!=len(set(ids)):errors.append("Doppelte Baustein-IDs.")
    if set(ids)!=set(labels):errors.append("Bausteine fehlen oder passen nicht zum gewählten Störungsort.")
    if len(pairs)!=len(set(pairs)):errors.append("Doppelte Verbindungen entfernen.")
    if set(pairs)!=expected:errors.append("Signalverbindungen entsprechen noch nicht dem geschlossenen Regelkreis inklusive Rückführung und Störung.")
    for node in state.nodes:
        if node.id not in {"soll","stoerung"} and node.type!="default":errors.append("Ein-/Ausgangsanschlüsse eines Bausteins sind unvollständig.")
    return errors


def reset_graph():
    st.session_state.builder_graph=standard_graph(config_of("builder"))


def builder_view():
    owner="builder";c=config_of(owner)
    st.title("Regelkreis bauen")
    st.caption("Bausteine und Kanten sind der tatsächliche übertragene Signalweg. Modellparameter bleiben bei der Übernahme vollständig erhalten.")
    with st.sidebar:configuration_controls(owner)
    cols=st.columns(3)
    if cols[0].button("Bausteine mit Standardverbindungen einsetzen"):
        reset_graph()
    if cols[1].button("Nur Bausteine einsetzen"):
        state=standard_graph(c);state.edges=[];st.session_state.builder_graph=state
    if cols[2].button("Aufbau leeren"):
        st.session_state.builder_graph=StreamlitFlowState([],[])
    state=st.session_state.builder_graph
    if state.nodes:
        state=streamlit_flow("lab_builder_canvas",state,height=460,fit_view=True,allow_new_edges=True,
                enable_node_menu=True,enable_edge_menu=True,get_node_on_click=True,get_edge_on_click=True)
        st.session_state.builder_graph=state
    else:st.info("Setze zunächst die Bausteine ein. Danach lassen sich die Anschlüsse verbinden und Kanten über ihr Kontextmenü entfernen.")
    errors=validate_graph(c,state)
    for err in errors:st.warning(err)
    model_errors=show_diagnostics(c)
    if not errors and not model_errors:st.success("Aktueller Aufbau vollständig verbunden und Eingaben gültig.")
    # Never reuse an earlier approval after a graph or parameter edit.
    if st.button("Geprüften Aufbau in Simulation übernehmen",disabled=bool(errors or model_errors)):
        fresh_errors=validate_graph(config_of(owner),st.session_state.builder_graph)
        _,fresh_model_errors,_,_=inspect_config(config_of(owner))
        if fresh_errors or fresh_model_errors:st.error("Der Aufbau hat sich geändert. Bitte erneut prüfen.")
        else:transfer_config(owner,"simulation");st.rerun()
    st.button("Aktuelle Simulation hier importieren",on_click=transfer_config,args=("simulation","builder"))


def main():
    st.set_page_config(page_title="Regelkreis-Labor",page_icon="🔧",layout="wide")
    init_session()
    views=["Start","Simulation","Praxisvorgaben","Regelkreis bauen"]
    for col,view in zip(st.columns(4),views):
        if col.button(view,key="nav_"+view,width="stretch"):st.session_state.view=view;st.rerun()
    view=st.session_state.view
    if view=="Start":
        st.title("Regelkreis-Labor")
        st.write("Regelkreise aufbauen, physikalische Bilanzen untersuchen und Anforderungen an der berechneten Antwort prüfen.")
        st.info("Die 70 Praxisvorgaben sind nachvollziehbare Beispiele. Komplexe Anlagen ohne passendes Bilanzmodell sind als Konzeptvorlagen gekennzeichnet; nicht implementierte Mehrkreisstrategien werden gesperrt.")
        a,b=st.columns(2)
        if a.button("Freie Simulation öffnen",type="primary"):st.session_state.view="Simulation";st.rerun()
        if b.button("Praxisvorgaben auswählen"):st.session_state.view="Praxisvorgaben";st.rerun()
        st.caption(f"Version {APP_VERSION} · SI-Bilanzen, begrenzter Rechenaufwand und gemeinsame Eingabeprüfung.")
    elif view=="Regelkreis bauen":builder_view()
    else:
        owner="physical" if view=="Praxisvorgaben" else "simulation"
        st.title(view)
        with st.sidebar:configuration_controls(owner)
        c=config_of(owner)
        col1,col2=st.columns(2)
        if owner=="physical":col1.button("In Simulation übernehmen",on_click=transfer_config,args=(owner,"simulation"))
        col2.button("In Regelkreis-Aufbau übernehmen",on_click=transfer_config,args=(owner,"builder"))
        show_result(owner,c)
        if owner=="physical":
            with st.expander("Signalbild"):
                graph_key="lab_physical_graph_"+fingerprint({"kind":c["kind"],"disturbance":c["disturbance_position"]})[:12]
                if graph_key not in st.session_state:st.session_state[graph_key]=standard_graph(c)
                graph=streamlit_flow(graph_key+"_canvas",st.session_state[graph_key],height=300,fit_view=True,allow_new_edges=False)
                st.session_state[graph_key]=graph


if __name__ == "__main__":
    main()
