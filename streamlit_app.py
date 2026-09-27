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


# ------------------------------------------------------------
# Seiteneinstellungen
# ------------------------------------------------------------

st.set_page_config(
    page_title="Regelkreis-Labor",
    layout="wide"
)

# ------------------------------------------------------------
# Grundzustand für Navigation
# ------------------------------------------------------------

if "app_started" not in st.session_state:
    st.session_state.app_started = False

if "active_view" not in st.session_state:
    st.session_state.active_view = "simulation"


# ------------------------------------------------------------
# Obere Navigation
# ------------------------------------------------------------

def render_top_navigation():
    """
    Horizontale Arbeitsbereich-Navigation oberhalb der App.
    Diese Navigation wird auf jeder Ansicht angezeigt.
    """

    with st.container(border=True, key="top_navigation"):
        col1, col2, col3, col4 = st.columns([1.2, 1.6, 1.7, 1.5])

        with col1:
            if st.button(
                "Simulation",
                width="stretch",
                type=(
                    "primary"
                    if st.session_state.app_started
                    and st.session_state.active_view == "simulation"
                    else "secondary"
                )
            ):
                st.session_state.app_started = True
                st.session_state.active_view = "simulation"
                st.rerun()

        with col2:
            if st.button(
                "Physikalischer Wirkplan-Builder",
                width="stretch",
                type="primary" if st.session_state.active_view == "wirkplan" else "secondary"
            ):
                st.session_state.app_started = True
                st.session_state.active_view = "wirkplan"
                st.rerun()

        with col3:
            if st.button(
                "Visueller Regelkreis-Builder",
                width="stretch",
                type="primary" if st.session_state.active_view == "builder" else "secondary"
            ):
                st.session_state.app_started = True
                st.session_state.active_view = "builder"
                st.rerun()

        with col4:
            if st.button(
                "Startformular neu öffnen",
                width="stretch",
                type="primary" if not st.session_state.app_started else "secondary"
            ):
                st.session_state.app_started = False
                st.session_state.active_view = "start"
                st.rerun()


render_top_navigation()

# ------------------------------------------------------------
# Startformular / Regelkreis-Assistent
# ------------------------------------------------------------

def get_default_parameters(
    lernziel: str,
    controller_type: str,
    plant_type: str,
    disturbance_position: str,
    schwierigkeitsgrad: str
):
    defaults = {
        "kp": 2.0,
        "ki": 0.5,
        "kd": 0.0,
        "ks": 1.0,
        "ts": 2.0,
        "zeta": 0.7,
        "omega0": 2.0,
        "setpoint": 1.0,
        "t_end": 20.0,
        "dt": 0.01,
        "disturbance_time": 8.0,
        "disturbance_value": -0.3,
    }

    if controller_type == "P":
        defaults["kp"] = 2.0
        defaults["ki"] = 0.0
        defaults["kd"] = 0.0

    elif controller_type == "PI":
        defaults["kp"] = 2.0
        defaults["ki"] = 0.5
        defaults["kd"] = 0.0

    elif controller_type == "PID":
        defaults["kp"] = 3.0
        defaults["ki"] = 0.5
        defaults["kd"] = 0.2

    if plant_type == "PT1":
        defaults["ts"] = 2.0
        defaults["zeta"] = 0.7
        defaults["omega0"] = 2.0

    elif plant_type == "PT2":
        defaults["ts"] = 2.0
        defaults["zeta"] = 0.6
        defaults["omega0"] = 2.0

    if disturbance_position == "Keine Störung":
        defaults["disturbance_time"] = 0.0
        defaults["disturbance_value"] = 0.0

    elif disturbance_position in ["Vor der Strecke", "Am Ausgang"]:
        defaults["disturbance_time"] = 8.0
        defaults["disturbance_value"] = -0.3

    if lernziel == "Grundverhalten verstehen":
        defaults["t_end"] = 20.0
        defaults["dt"] = 0.01

    elif lernziel == "Störverhalten untersuchen":
        defaults["t_end"] = 30.0
        defaults["disturbance_time"] = 10.0

        if disturbance_position == "Keine Störung":
            defaults["disturbance_value"] = 0.0
        else:
            defaults["disturbance_value"] = -0.3

    elif lernziel == "Regler optimieren":
        defaults["t_end"] = 30.0
        defaults["dt"] = 0.005

    if schwierigkeitsgrad == "Einsteiger":
        defaults["dt"] = 0.01

    elif schwierigkeitsgrad == "Fortgeschritten":
        defaults["dt"] = 0.005

    elif schwierigkeitsgrad == "Experte":
        defaults["dt"] = 0.002

    return defaults


# ------------------------------------------------------------
# Robuste Standardwerte für Session-State
# ------------------------------------------------------------
# Diese Werte werden benötigt, wenn der Benutzer z. B. direkt über die
# obere Navigation in den Builder wechselt und anschließend in die
# Simulation zurückkehrt, ohne das Startformular erneut abzusenden.

if "lernziel" not in st.session_state:
    st.session_state.lernziel = "Grundverhalten verstehen"

if "schwierigkeitsgrad" not in st.session_state:
    st.session_state.schwierigkeitsgrad = "Fortgeschritten"

if "controller_type" not in st.session_state:
    st.session_state.controller_type = "PI"

if "plant_type" not in st.session_state:
    st.session_state.plant_type = "PT1"

if "disturbance_position" not in st.session_state:
    st.session_state.disturbance_position = "Keine Störung"

if "defaults" not in st.session_state:
    st.session_state.defaults = get_default_parameters(
        lernziel=st.session_state.lernziel,
        controller_type=st.session_state.controller_type,
        plant_type=st.session_state.plant_type,
        disturbance_position=st.session_state.disturbance_position,
        schwierigkeitsgrad=st.session_state.schwierigkeitsgrad
    )


if not st.session_state.app_started:

    st.title("Regelkreis-Assistent")

    st.markdown(
        """
        Diese App unterstützt dich beim Aufbau und bei der Analyse eines geschlossenen Regelkreises.

        Wähle zuerst aus, was du untersuchen möchtest. Danach erstellt die App automatisch einen passenden
        Regelkreis mit sinnvollen Startparametern.
        """
    )

    st.subheader("1. Arbeitsbereich auswählen")
    arbeitsbereich = st.radio(
        "Wo möchtest du nach dem Start weiterarbeiten?",
        [
            "Simulation",
            "Physikalischer Wirkplan-Builder",
            "Visueller Regelkreis-Builder",
        ],
        horizontal=True,
        help=(
            "Simulation öffnet direkt die Kurvenberechnung. Der physikalische Builder startet "
            "mit einer praxisnahen Anlage, der visuelle Builder mit einer frei verbindbaren Arbeitsfläche."
        ),
    )

    start_category = None
    start_variant = None
    start_actuator = None
    start_strategy = None
    if arbeitsbereich == "Physikalischer Wirkplan-Builder":
        start_category = st.selectbox(
            "Anlagenart / Gewerk",
            list(PRACTICAL_PROCESS_CATALOG),
            help="Legt den fachlichen Anlagenbereich und die danach angebotenen Regelungsvarianten fest.",
        )
        start_variants = PRACTICAL_PROCESS_CATALOG[start_category]
        start_variant = st.selectbox(
            "Regelungsvariante",
            list(start_variants),
            help="Wählt die konkrete Regelaufgabe und damit passende Prozess-, Sensor- und Störungswerte.",
        )
        start_profile = start_variants[start_variant]
        start_actuator = st.selectbox(
            "Stellglied / Antrieb",
            start_profile["actuators"],
            help="Das Stellglied setzt das Ausgangssignal des Reglers physikalisch um.",
        )
        start_strategy = st.selectbox(
            "Regelstrategie",
            start_profile["strategies"],
            help="Bestimmt die fachliche Regelstrategie und ihre Abbildung im Simulator.",
        )

    with st.form("start_formular"):

        st.subheader("2. Ziel der Untersuchung")

        lernziel = st.selectbox(
            "Was möchtest du mit dem Regelkreis untersuchen?",
            [
                "Grundverhalten verstehen",
                "Störverhalten untersuchen",
                "Regler optimieren"
            ],
            help=(
                "Das Lernziel legt fest, worauf die App den Schwerpunkt setzt: "
                "Grundverhalten, Störverhalten oder gezielte Regleroptimierung."
            )
        )

        with st.expander("Hilfe: Was bedeuten die Lernziele?", expanded=False):
            st.markdown(
                """
                **Grundverhalten verstehen**  
                Die App zeigt das grundsätzliche Verhalten eines geschlossenen Regelkreises.  
                Du erkennst, wie Sollwert, Regelgröße, Regeldifferenz, Regler, Strecke und Rückführung zusammenwirken.

                **Störverhalten untersuchen**  
                Die App legt den Schwerpunkt auf Störungen.  
                Du kannst erkennen, wie der Regelkreis auf eine Störung reagiert und ob der Regler diese wieder ausregelt.

                **Regler optimieren**  
                Die App ist stärker auf Parametervergleich ausgelegt.  
                Du kannst untersuchen, wie sich Kp, Ki und Kd auf Überschwingen, Einschwingzeit und bleibende Regelabweichung auswirken.
                """
            )

        st.subheader("3. Aufbau des Regelkreises")

        controller_type = st.selectbox(
            "Welcher Reglertyp soll verwendet werden?",
            ["P", "PI", "PID"],
            index=1
        )

        plant_type = st.selectbox(
            "Welche Strecke soll untersucht werden?",
            ["PT1", "PT2"],
            index=0
        )

        disturbance_position = st.selectbox(
            "Soll eine Störung berücksichtigt werden?",
            [
                "Keine Störung",
                "Vor der Strecke",
                "Am Ausgang"
            ],
            index=0
        )

        st.subheader("4. Bedienmodus")

        schwierigkeitsgrad = st.radio(
            "Wie viele Details möchtest du einstellen können?",
            [
                "Einsteiger",
                "Fortgeschritten",
                "Experte"
            ],
            horizontal=True,
            help=(
                "Der Modus steuert, wie viele technische Einstellmöglichkeiten sichtbar sind "
                "und welche Werte die App automatisch vorbelegt."
            )
        )

        with st.expander("Hilfe: Was bedeuten die Modi?", expanded=False):
            st.markdown(
                """
                **Einsteiger**  
                Die App übernimmt viele technische Werte automatisch.  
                Geeignet, wenn du erstmal das Prinzip des Regelkreises verstehen möchtest.

                **Fortgeschritten**  
                Die App rechnet mit feineren Standardwerten und eignet sich besser zum Vergleichen von Regelverhalten.  
                Die Bedienung bleibt aber weiterhin geführt.

                **Experte**  
                Du bekommst zusätzliche technische Einstellmöglichkeiten, zum Beispiel die Simulations-Schrittweite `dt`.  
                Dieser Modus ist sinnvoll, wenn du gezielt Parameter untersuchen möchtest.
                """
            )

        submitted = st.form_submit_button("Regelkreis erstellen")

    if submitted:
        defaults = get_default_parameters(
            lernziel=lernziel,
            controller_type=controller_type,
            plant_type=plant_type,
            disturbance_position=disturbance_position,
            schwierigkeitsgrad=schwierigkeitsgrad
        )

        st.session_state.app_started = True
        st.session_state.lernziel = lernziel
        st.session_state.controller_type = controller_type
        st.session_state.plant_type = plant_type
        st.session_state.disturbance_position = disturbance_position
        st.session_state.schwierigkeitsgrad = schwierigkeitsgrad
        st.session_state.defaults = defaults

        st.session_state.builder_config = defaults.copy()
        st.session_state.builder_config["controller_type"] = controller_type
        st.session_state.builder_config["plant_type"] = plant_type
        st.session_state.builder_config["disturbance_position"] = disturbance_position

        if arbeitsbereich == "Physikalischer Wirkplan-Builder":
            st.session_state.pending_start_wirkplan = {
                "anlagenart": start_category,
                "regelungsvariante": start_variant,
                "stellglied_typ": start_actuator,
                "regelstrategie": start_strategy,
            }
            st.session_state.active_view = "wirkplan"
        elif arbeitsbereich == "Visueller Regelkreis-Builder":
            st.session_state.builder_step = 1
            st.session_state.builder_last_validation = None
            st.session_state.pop("builder_flow_state", None)
            st.session_state.active_view = "builder"
        else:
            st.session_state.active_view = "simulation"

        st.rerun()

    st.stop()


# ------------------------------------------------------------
# Grundzustände
# ------------------------------------------------------------


if "builder_config" not in st.session_state:
    st.session_state.builder_config = {
        "controller_type": st.session_state.get("controller_type", "PI"),
        "plant_type": st.session_state.get("plant_type", "PT1"),
        "disturbance_position": st.session_state.get("disturbance_position", "Keine Störung"),
        "kp": 2.0,
        "ki": 0.5,
        "kd": 0.0,
        "ks": 1.0,
        "ts": 2.0,
        "zeta": 0.7,
        "omega0": 2.0,
        "setpoint": 1.0,
        "t_end": 20.0,
        "dt": 0.01,
        "disturbance_time": 8.0,
        "disturbance_value": -0.3,
    }

if "builder_step" not in st.session_state:
    st.session_state.builder_step = 1

if "builder_last_validation" not in st.session_state:
    st.session_state.builder_last_validation = None

# builder_flow_state wird bewusst erst beim Öffnen des visuellen Builders
# initialisiert, damit streamlit-flow den Zustand persistent halten kann.


if "wirkplan_config" not in st.session_state:
    st.session_state.wirkplan_config = {
        "anlagenart": "RLT / Lüftung",
        "regelungsvariante": "Zulufttemperatur",
        "stellglied_typ": "2-Wege-Regelventil",
        "regelstrategie": "Automatische Empfehlung",
        "prozessart": "Temperaturregelung",
        "stellgroesse": "Heizleistung P [W]",
        "prozessglied": "thermische Strecke / Raum",
        "speicher": "thermische Masse",
        "regelgroesse": "Temperatur T [°C]",
        "stoergroesse": "Außentemperatur / Fremdwärme",
        "traegheit": "träge",
        "ueberschwingen_zulaessig": "Nein",
        "bleibende_abweichung_erlaubt": "Nein",
        "stoerungen_relevant": "Ja",
        "sensor": "Temperaturfühler",
        "messglied": "Messumformer",
        "stellglied": "Leistungssteller / Heizung",
        "sollwertgeber": "Sollwertvorgabe",
        "messbereich_min": 0.0,
        "messbereich_max": 100.0,
        "sensor_zeitkonstante_s": 0.2,
        "messrauschen": 0.0,
        "totzeit_s": 0.0,
        "stellgroesse_min": 0.0,
        "stellgroesse_max": 100.0,
        "stellrate_max": 100.0,
        "auslegung": "Automatisch",
        "man_controller_type": "PI",
        "man_plant_type": "PT1",
        "man_kp": 2.0,
        "man_ki": 0.4,
        "man_kd": 0.0,
        "man_ks": 1.0,
        "man_ts": 3.0,
        "man_zeta": 0.7,
        "man_omega0": 2.0,
        "man_setpoint": 1.0,
        "man_t_end": 25.0,
        "man_dt": 0.01,
        "stoerort": "Vor der Strecke",
        "stoerzeit_s": 10.0,
        "stoerwert": -0.3,
        "reale_daten_aktiv": True,
        "eingabetiefe": "Einfach",
        # Temperaturstrecke
        "temp_betriebsart": "Heizen",
        "temp_medium": "Wasser",
        "temp_volumen_m3": 1.0,
        "temp_heizleistung_kw": 12.0,
        "temp_umgebung_c": 15.0,
        "temp_soll_c": 55.0,
        "temp_waermeverlust_w_k": 180.0,
        "temp_wirkungsgrad": 0.95,
        "temp_dichte_kg_m3": 998.0,
        "temp_cp_kj_kgk": 4.18,
        # Drehzahlstrecke
        "motor_leistung_kw": 5.5,
        "motor_nenndrehzahl_rpm": 1450.0,
        "motor_soll_rpm": 1200.0,
        "motor_hochlaufzeit_s": 5.0,
        "motor_leitung_m": 50.0,
        "motor_querschnitt_mm2": 2.5,
        "motor_spannung_v": 400.0,
        "motor_wirkungsgrad": 0.88,
        "motor_leistungsfaktor": 0.82,
        "motor_traegheit_kgm2": 0.18,
        "motor_lastmoment_nm": 20.0,
        # Füllstandsstrecke
        "tank_form": "Zylindrisch",
        "tank_volumen_m3": 2.5,
        "tank_hoehe_m": 2.0,
        "tank_zulauf_m3h": 8.0,
        "tank_abfluss_m3h": 3.0,
        "tank_soll_m": 1.4,
        # Druckstrecke
        "druck_volumen_m3": 1.0,
        "druck_max_bar": 10.0,
        "druck_soll_bar": 6.0,
        "druck_foerderstrom_m3h": 60.0,
        "druck_verbrauch_m3h": 20.0,
        "druck_zeitkonstante_s": 3.0,
        # Durchflussstrecke
        "flow_max_m3h": 100.0,
        "flow_soll_m3h": 60.0,
        "flow_rohrlaenge_m": 20.0,
        "flow_durchmesser_mm": 80.0,
        "flow_ventilzeit_s": 0.8,
        "flow_druckverlust_bar": 1.5,
        # Positionsstrecke
        "pos_masse_kg": 25.0,
        "pos_feder_n_m": 1200.0,
        "pos_daempfung_ns_m": 180.0,
        "pos_stellkraft_n": 1500.0,
        "pos_hub_mm": 500.0,
        "pos_soll_mm": 250.0,
        # Generische Strecke für Prozess-, Chemie- und Energieregelungen
        "generic_plant_type": "PT1",
        "generic_ks": 1.0,
        "generic_ts": 10.0,
        "generic_zeta": 0.7,
        "generic_omega0": 1.0,
        "generic_setpoint": 1.0,
        "generic_unit": "Prozesseinheit",
        "generic_t_end": 60.0,
    }


# ------------------------------------------------------------
# Hilfsfunktionen Simulation
# ------------------------------------------------------------

def blockdiagramm(controller_type: str, plant_type: str, disturbance_position: str) -> str:
    disturbance_label = "Störung d(t)"

    if disturbance_position == "Keine Störung":
        disturbance_part = ""
        input_to_plant = "regler -> strecke"
        output_label = "strecke -> ausgang"

    elif disturbance_position == "Vor der Strecke":
        disturbance_part = f"""
        dist [label="{disturbance_label}", shape=ellipse, style=dashed];
        summ2 [label="Σ", shape=circle, fillcolor="#FFFFFF"];
        regler -> summ2;
        dist -> summ2;
        summ2 -> strecke;
        """

        input_to_plant = ""
        output_label = "strecke -> ausgang"

    else:
        disturbance_part = f"""
        dist [label="{disturbance_label}", shape=ellipse, style=dashed];
        summ3 [label="Σ", shape=circle, fillcolor="#FFFFFF"];
        strecke -> summ3;
        dist -> summ3;
        summ3 -> ausgang;
        """

        input_to_plant = "regler -> strecke"
        output_label = ""

    return f"""
    digraph {{
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#F7F7F7", fontname="Arial"];

        soll [label="Sollwert w(t)"];
        summ1 [label="Σ", shape=circle, fillcolor="#FFFFFF"];
        regler [label="{controller_type}-Regler"];
        strecke [label="{plant_type}-Strecke"];
        ausgang [label="Regelgröße y(t)"];
        rueck [label="Rückführung", shape=box];

        soll -> summ1;
        summ1 -> regler [label="e(t)"];
        {input_to_plant};
        {output_label};
        ausgang -> rueck;
        rueck -> summ1 [label="-y(t)"];

        {disturbance_part}
    }}
    """


def simulate_control_loop(
    controller_type: str,
    plant_type: str,
    kp: float,
    ki: float,
    kd: float,
    ks: float,
    ts: float,
    zeta: float,
    omega0: float,
    setpoint: float,
    t_end: float,
    dt: float,
    disturbance_position: str,
    disturbance_time: float,
    disturbance_value: float,
    u_min=None,
    u_max=None,
):
    t = np.arange(0.0, t_end + dt, dt)

    y_plant = np.zeros_like(t)
    y_out = np.zeros_like(t)
    u_controller = np.zeros_like(t)
    error = np.zeros_like(t)
    disturbance = np.zeros_like(t)

    integral_error = 0.0
    previous_error = 0.0
    velocity = 0.0

    for k in range(1, len(t)):

        if t[k] >= disturbance_time:
            disturbance[k] = disturbance_value
        else:
            disturbance[k] = 0.0

        error[k] = setpoint - y_out[k - 1]

        p_part = kp * error[k]

        previous_integral = integral_error

        if controller_type in ["PI", "PID"]:
            integral_error += error[k] * dt
        else:
            integral_error = 0.0

        i_part = ki * integral_error

        if controller_type == "PID":
            d_part = kd * (error[k] - previous_error) / dt
        else:
            d_part = 0.0

        raw_output = p_part + i_part + d_part
        limited_output = raw_output
        if u_min is not None:
            limited_output = max(float(u_min), limited_output)
        if u_max is not None:
            limited_output = min(float(u_max), limited_output)

        # Einfaches Anti-Windup: Wenn die Begrenzung aktiv ist und der Fehler
        # weiter in die Sättigung treibt, wird der letzte Integrationsschritt verworfen.
        if limited_output != raw_output and controller_type in ["PI", "PID"]:
            drives_further_into_limit = (
                (u_max is not None and raw_output > float(u_max) and error[k] > 0)
                or (u_min is not None and raw_output < float(u_min) and error[k] < 0)
            )
            if drives_further_into_limit:
                integral_error = previous_integral
                i_part = ki * integral_error
                raw_output = p_part + i_part + d_part
                limited_output = raw_output
                if u_min is not None:
                    limited_output = max(float(u_min), limited_output)
                if u_max is not None:
                    limited_output = min(float(u_max), limited_output)

        u_controller[k] = limited_output
        previous_error = error[k]

        if disturbance_position == "Vor der Strecke":
            u_effective = u_controller[k] + disturbance[k]
            output_disturbance = 0.0

        elif disturbance_position == "Am Ausgang":
            u_effective = u_controller[k]
            output_disturbance = disturbance[k]

        else:
            u_effective = u_controller[k]
            output_disturbance = 0.0

        if plant_type == "PT1":
            dy = (ks * u_effective - y_plant[k - 1]) / ts
            y_plant[k] = y_plant[k - 1] + dy * dt

        else:
            acceleration = (
                ks * omega0**2 * u_effective
                - 2 * zeta * omega0 * velocity
                - omega0**2 * y_plant[k - 1]
            )

            velocity = velocity + acceleration * dt
            y_plant[k] = y_plant[k - 1] + velocity * dt

        y_out[k] = y_plant[k] + output_disturbance

    df = pd.DataFrame({
        "Zeit [s]": t,
        "Sollwert w": setpoint,
        "Regelgröße y": y_out,
        "Stellgröße u": u_controller,
        "Regeldifferenz e": error,
        "Störung d": disturbance,
    })

    return df


def calculate_metrics(df: pd.DataFrame, setpoint: float):
    y = df["Regelgröße y"].to_numpy()
    t = df["Zeit [s]"].to_numpy()

    final_value = y[-1]
    steady_error = setpoint - final_value

    if setpoint != 0:
        overshoot = max(0.0, (np.max(y) - setpoint) / abs(setpoint) * 100)
        tolerance = 0.02 * abs(setpoint)
    else:
        overshoot = 0.0
        tolerance = 0.02

    settling_time = None

    for i in range(len(y)):
        if np.all(np.abs(y[i:] - setpoint) <= tolerance):
            settling_time = t[i]
            break

    return final_value, steady_error, overshoot, settling_time



# ------------------------------------------------------------
# Visueller Regelkreis-Builder – interaktiver Engineering-Baukasten
# ------------------------------------------------------------

def default_builder_config():
    """Standardwerte für einen neuen visuellen Regelkreis."""
    return {
        "controller_type": "PI",
        "plant_type": "PT1",
        "disturbance_position": "Keine Störung",
        "kp": 2.0,
        "ki": 0.5,
        "kd": 0.0,
        "ks": 1.0,
        "ts": 2.0,
        "zeta": 0.7,
        "omega0": 2.0,
        "setpoint": 1.0,
        "t_end": 20.0,
        "dt": 0.01,
        "disturbance_time": 8.0,
        "disturbance_value": -0.3,
        "setpoint_name": "Sollwert w(t)",
        "output_name": "Regelgröße y(t)",
    }


SHARED_PARAMETER_KEYS = (
    "kp", "ki", "kd", "ks", "ts", "zeta", "omega0", "setpoint",
    "t_end", "dt", "disturbance_time", "disturbance_value", "u_min", "u_max",
)


def shared_parameters_from_session():
    """Liest den gemeinsamen Parametersatz aus der Simulation."""
    values = default_builder_config()
    values.update(st.session_state.get("defaults", {}))
    values.update({
        "controller_type": st.session_state.get("controller_type", "PI"),
        "plant_type": st.session_state.get("plant_type", "PT1"),
        "disturbance_position": st.session_state.get(
            "disturbance_position", "Keine Störung"
        ),
    })
    return values


def shared_parameters_from_derived(derived: dict):
    """Normalisiert die Ableitung des physikalischen Builders für alle Ansichten."""
    physical = derived.get("physical", {})
    values = {
        "controller_type": derived["controller_type"],
        "plant_type": derived["plant_type"],
        "disturbance_position": derived["disturbance_position"],
    }
    for key in SHARED_PARAMETER_KEYS:
        if key in derived:
            values[key] = derived[key]
    if physical.get("active"):
        values["u_min"] = physical.get("u_min")
        values["u_max"] = physical.get("u_max")
    return values


def apply_shared_parameters(parameters: dict, source_label: str):
    """Verteilt einen Parametersatz an Simulation und beide Builder."""
    values = default_builder_config()
    values.update(parameters)

    controller_type = values.get("controller_type", "PI")
    plant_type = values.get("plant_type", "PT1")
    disturbance_position = values.get("disturbance_position", "Keine Störung")
    st.session_state.controller_type = controller_type
    st.session_state.plant_type = plant_type
    st.session_state.disturbance_position = disturbance_position

    defaults = {
        key: values.get(key)
        for key in SHARED_PARAMETER_KEYS
        if values.get(key) is not None
    }
    st.session_state.defaults = defaults

    builder_config = st.session_state.get("builder_config", default_builder_config()).copy()
    builder_config.update(values)
    st.session_state.builder_config = builder_config

    wirkplan = st.session_state.get("wirkplan_config", {}).copy()
    wirkplan.update({
        "auslegung": "Manuell",
        "reale_daten_aktiv": False,
        "man_controller_type": controller_type,
        "man_plant_type": plant_type,
        "man_kp": float(values.get("kp", 2.0)),
        "man_ki": float(values.get("ki", 0.5)),
        "man_kd": float(values.get("kd", 0.0)),
        "man_ks": float(values.get("ks", 1.0)),
        "man_ts": float(values.get("ts", 2.0)),
        "man_zeta": float(values.get("zeta", 0.7)),
        "man_omega0": float(values.get("omega0", 2.0)),
        "man_setpoint": float(values.get("setpoint", 1.0)),
        "man_t_end": float(values.get("t_end", 20.0)),
        "man_dt": float(values.get("dt", 0.01)),
        "stoerungen_relevant": (
            "Nein" if disturbance_position == "Keine Störung" else "Ja"
        ),
        "stoerort": (
            disturbance_position
            if disturbance_position in ["Vor der Strecke", "Am Ausgang"]
            else "Vor der Strecke"
        ),
        "stoerzeit_s": float(values.get("disturbance_time", 0.0)),
        "stoerwert": float(values.get("disturbance_value", 0.0)),
    })
    st.session_state.wirkplan_config = wirkplan

    # Erst im folgenden Rerun löschen: Streamlit erlaubt keine Mutation
    # eines Widget-Keys, nachdem das betreffende Widget gerendert wurde.
    st.session_state.clear_parameter_widgets_pending = True

    st.session_state.parameter_sync_notice = (
        f"Parameter aus {source_label} wurden an alle drei Arbeitsbereiche übertragen."
    )


def reset_shared_parameters():
    """Setzt die gemeinsamen Simulationsparameter auf den Programmstandard."""
    standard = default_builder_config()
    standard.update({"u_min": None, "u_max": None})
    apply_shared_parameters(standard, "den Standardwerten")


def clear_pending_parameter_widgets():
    """Entfernt veraltete Widgetwerte vor dem Rendern der jeweiligen Ansicht."""
    if not st.session_state.pop("clear_parameter_widgets_pending", False):
        return
    for key in list(st.session_state.keys()):
        if key.startswith("ib_") or key.startswith("sim_"):
            del st.session_state[key]
    for key in [
        "wirkplan_auslegung", "wirkplan_reale_daten_aktiv",
        "wirkplan_man_controller_type", "wirkplan_man_plant_type",
        "wirkplan_man_kp", "wirkplan_man_ki", "wirkplan_man_kd",
        "wirkplan_man_ks", "wirkplan_man_ts", "wirkplan_man_zeta",
        "wirkplan_man_omega0", "wirkplan_man_setpoint",
        "wirkplan_man_t_end", "wirkplan_man_dt", "wirkplan_stoerungen",
        "wirkplan_stoerort", "wirkplan_stoerzeit_s", "wirkplan_stoerwert",
    ]:
        st.session_state.pop(key, None)


def reset_interactive_builder():
    """Setzt Arbeitsfläche, Parameter und Engineering-Schritt zurück."""
    st.session_state.builder_step = 1
    st.session_state.builder_last_validation = None
    st.session_state.builder_config = default_builder_config()

    if "builder_flow_state" in st.session_state:
        del st.session_state.builder_flow_state

    # Widget-Zustände des Builders entfernen, damit beim Neustart
    # wieder die Standardwerte angezeigt werden.
    for key in list(st.session_state.keys()):
        if key.startswith("ib_"):
            del st.session_state[key]


def make_builder_node(kind: str, config: dict):
    """Erzeugt genau einen regelungstechnischen Baustein."""
    node_specs = {
        "setpoint": {
            "id": "sollwert",
            "pos": (20, 220),
            "content": f"{config.get('setpoint_name', 'Sollwert w(t)')}<br>w = {config.get('setpoint', 1.0)}",
            "node_type": "input",
            "source_position": "right",
            "target_position": None,
        },
        "sum": {
            "id": "vergleich",
            "pos": (210, 220),
            "content": "Vergleichsstelle Σ<br>e = w − y",
            "node_type": "default",
            "source_position": "right",
            "target_position": "left",
        },
        "controller": {
            "id": "regler",
            "pos": (420, 220),
            "content": (
                f"{config.get('controller_type', 'PI')}-Regler<br>"
                f"Kp={config.get('kp', 2.0)}<br>"
                f"Ki={config.get('ki', 0.5)}<br>"
                f"Kd={config.get('kd', 0.0)}"
            ),
            "node_type": "default",
            "source_position": "right",
            "target_position": "left",
        },
        "plant": {
            "id": "strecke",
            "pos": (640, 220),
            "content": (
                f"PT1-Strecke<br>Ks={config.get('ks', 1.0)}<br>Ts={config.get('ts', 2.0)} s"
                if config.get("plant_type", "PT1") == "PT1"
                else (
                    f"PT2-Strecke<br>Ks={config.get('ks', 1.0)}<br>"
                    f"ζ={config.get('zeta', 0.7)}<br>"
                    f"ω0={config.get('omega0', 2.0)} rad/s"
                )
            ),
            "node_type": "default",
            "source_position": "right",
            "target_position": "left",
        },
        "output": {
            "id": "ausgang",
            "pos": (860, 220),
            "content": config.get("output_name", "Regelgröße y(t)"),
            # default statt output, damit der Ausgang auch eine Quelle
            # für die Rückführung besitzen kann.
            "node_type": "default",
            "source_position": "right",
            "target_position": "left",
        },
        "feedback": {
            "id": "rueckfuehrung",
            "pos": (470, 450),
            "content": "Rückführung<br>Istwert y(t)",
            "node_type": "default",
            "source_position": "left",
            "target_position": "right",
        },
        "disturbance": {
            "id": "stoerung",
            "pos": (620, 40),
            "content": (
                f"Störung d(t)<br>d={config.get('disturbance_value', -0.3)}<br>"
                f"ab {config.get('disturbance_time', 8.0)} s"
            ),
            "node_type": "input",
            "source_position": "right",
            "target_position": None,
        },
        "dist_sum": {
            "id": "stoersumme",
            "pos": (650, 220),
            "content": "Stör-Summierstelle Σ",
            "node_type": "default",
            "source_position": "right",
            "target_position": "left",
        },
    }

    spec = node_specs[kind]
    data = {"content": spec["content"], "kind": kind}

    if kind == "controller":
        data.update({
            "controller_type": config.get("controller_type", "PI"),
            "kp": float(config.get("kp", 2.0)),
            "ki": float(config.get("ki", 0.5)),
            "kd": float(config.get("kd", 0.0)),
        })

    elif kind == "plant":
        data.update({
            "plant_type": config.get("plant_type", "PT1"),
            "ks": float(config.get("ks", 1.0)),
            "ts": float(config.get("ts", 2.0)),
            "zeta": float(config.get("zeta", 0.7)),
            "omega0": float(config.get("omega0", 2.0)),
        })

    elif kind == "setpoint":
        data.update({
            "setpoint_name": config.get("setpoint_name", "Sollwert w(t)"),
            "setpoint": float(config.get("setpoint", 1.0)),
        })

    elif kind == "output":
        data.update({"output_name": config.get("output_name", "Regelgröße y(t)")})

    elif kind == "disturbance":
        data.update({
            "disturbance_position": config.get("disturbance_position", "Vor der Strecke"),
            "disturbance_time": float(config.get("disturbance_time", 8.0)),
            "disturbance_value": float(config.get("disturbance_value", -0.3)),
        })

    kwargs = dict(
        id=spec["id"],
        pos=spec["pos"],
        data=data,
        node_type=spec["node_type"],
        source_position=spec["source_position"],
        draggable=True,
    )

    if spec["target_position"] is not None:
        kwargs["target_position"] = spec["target_position"]

    return StreamlitFlowNode(**kwargs)


def init_builder_flow_state():
    """
    Initialisiert die Arbeitsfläche genau einmal.
    streamlit-flow 1.6.x benötigt einen persistenten State im session_state.
    """
    if "builder_flow_state" not in st.session_state:
        st.session_state.builder_flow_state = StreamlitFlowState([], [])


def get_builder_node(kind: str):
    """Liefert den ersten Baustein eines Typs oder None."""
    init_builder_flow_state()
    for node in st.session_state.builder_flow_state.nodes:
        if isinstance(node.data, dict) and node.data.get("kind") == kind:
            return node
    return None


def builder_has_node(kind: str) -> bool:
    return get_builder_node(kind) is not None


def add_builder_node(kind: str):
    """Fügt einen Baustein hinzu, sofern dieser noch nicht existiert."""
    init_builder_flow_state()

    if builder_has_node(kind):
        return False

    node = make_builder_node(kind, st.session_state.builder_config)
    st.session_state.builder_flow_state.nodes.append(node)
    return True


def remove_builder_node(kind: str):
    """Entfernt einen Baustein und alle daran hängenden Verbindungen."""
    init_builder_flow_state()

    ids = {
        node.id
        for node in st.session_state.builder_flow_state.nodes
        if isinstance(node.data, dict) and node.data.get("kind") == kind
    }

    if not ids:
        return

    st.session_state.builder_flow_state.nodes = [
        node for node in st.session_state.builder_flow_state.nodes
        if node.id not in ids
    ]

    st.session_state.builder_flow_state.edges = [
        edge for edge in st.session_state.builder_flow_state.edges
        if edge.source not in ids and edge.target not in ids
    ]


def sync_builder_nodes_from_config():
    """
    Synchronisiert Beschriftungen und Parameter der vorhandenen Bausteine
    mit den aktuellen Engineering-Einstellungen, ohne Positionen zu verändern.
    """
    init_builder_flow_state()
    config = st.session_state.builder_config

    for node in st.session_state.builder_flow_state.nodes:
        kind = node.data.get("kind") if isinstance(node.data, dict) else None

        if kind == "setpoint":
            node.data["setpoint_name"] = config.get("setpoint_name", "Sollwert w(t)")
            node.data["setpoint"] = float(config.get("setpoint", 1.0))
            node.data["content"] = (
                f"{config.get('setpoint_name', 'Sollwert w(t)')}<br>"
                f"w = {config.get('setpoint', 1.0)}"
            )

        elif kind == "output":
            node.data["output_name"] = config.get("output_name", "Regelgröße y(t)")
            node.data["content"] = config.get("output_name", "Regelgröße y(t)")

        elif kind == "controller":
            node.data["controller_type"] = config.get("controller_type", "PI")
            node.data["kp"] = float(config.get("kp", 2.0))
            node.data["ki"] = float(config.get("ki", 0.5))
            node.data["kd"] = float(config.get("kd", 0.0))
            node.data["content"] = (
                f"{config.get('controller_type', 'PI')}-Regler<br>"
                f"Kp={config.get('kp', 2.0)}<br>"
                f"Ki={config.get('ki', 0.5)}<br>"
                f"Kd={config.get('kd', 0.0)}"
            )

        elif kind == "plant":
            node.data["plant_type"] = config.get("plant_type", "PT1")
            node.data["ks"] = float(config.get("ks", 1.0))
            node.data["ts"] = float(config.get("ts", 2.0))
            node.data["zeta"] = float(config.get("zeta", 0.7))
            node.data["omega0"] = float(config.get("omega0", 2.0))

            if config.get("plant_type", "PT1") == "PT1":
                node.data["content"] = (
                    f"PT1-Strecke<br>Ks={config.get('ks', 1.0)}<br>"
                    f"Ts={config.get('ts', 2.0)} s"
                )
            else:
                node.data["content"] = (
                    f"PT2-Strecke<br>Ks={config.get('ks', 1.0)}<br>"
                    f"ζ={config.get('zeta', 0.7)}<br>"
                    f"ω0={config.get('omega0', 2.0)} rad/s"
                )

        elif kind == "disturbance":
            node.data["disturbance_position"] = config.get(
                "disturbance_position", "Vor der Strecke"
            )
            node.data["disturbance_time"] = float(config.get("disturbance_time", 8.0))
            node.data["disturbance_value"] = float(config.get("disturbance_value", -0.3))
            node.data["content"] = (
                f"Störung d(t)<br>d={config.get('disturbance_value', -0.3)}<br>"
                f"ab {config.get('disturbance_time', 8.0)} s"
            )

        elif kind == "dist_sum":
            if config.get("disturbance_position") == "Vor der Strecke":
                node.data["content"] = "Stör-Summierstelle Σ<br>u + d"
            elif config.get("disturbance_position") == "Am Ausgang":
                node.data["content"] = "Stör-Summierstelle Σ<br>y + d"
            else:
                node.data["content"] = "Stör-Summierstelle Σ"


def edge_exists(source: str, target: str) -> bool:
    """Prüft, ob eine gerichtete Verbindung existiert."""
    init_builder_flow_state()
    return any(
        edge.source == source and edge.target == target
        for edge in st.session_state.builder_flow_state.edges
    )


def allowed_builder_connections():
    """
    Liefert die Verbindungen, die von der aktuellen Simulation mathematisch
    unterstützt werden.

    Der Builder bleibt damit interaktiv, aber es können keine Verbindungen
    entstehen, die im hinterlegten Simulationsmodell keine Bedeutung haben.
    """
    config = st.session_state.builder_config
    disturbance_position = config.get("disturbance_position", "Keine Störung")

    allowed = {
        ("sollwert", "vergleich"),
        ("vergleich", "regler"),
        ("ausgang", "rueckfuehrung"),
        ("rueckfuehrung", "vergleich"),
    }

    if disturbance_position == "Keine Störung":
        allowed.update({
            ("regler", "strecke"),
            ("strecke", "ausgang"),
        })

    elif disturbance_position == "Vor der Strecke":
        allowed.update({
            ("regler", "stoersumme"),
            ("stoerung", "stoersumme"),
            ("stoersumme", "strecke"),
            ("strecke", "ausgang"),
        })

    elif disturbance_position == "Am Ausgang":
        allowed.update({
            ("regler", "strecke"),
            ("strecke", "stoersumme"),
            ("stoerung", "stoersumme"),
            ("stoersumme", "ausgang"),
        })

    return allowed


def sanitize_builder_edges():
    """
    Entfernt unzulässige, doppelte oder selbstbezügliche Verbindungen.

    Rückgabewert:
        dict mit Anzahl und Beschreibung der entfernten Verbindungen.
    """
    init_builder_flow_state()

    allowed = allowed_builder_connections()
    original_edges = list(st.session_state.builder_flow_state.edges)

    cleaned_edges = []
    seen_pairs = set()
    removed_invalid = 0
    removed_duplicates = 0
    removed_self = 0
    edge_options_changed = False

    for edge in original_edges:
        pair = (edge.source, edge.target)

        if edge.source == edge.target:
            removed_self += 1
            continue

        if pair not in allowed:
            removed_invalid += 1
            continue

        if pair in seen_pairs:
            removed_duplicates += 1
            continue

        if not getattr(edge, "deletable", False):
            edge.deletable = True
            edge_options_changed = True
        if not getattr(edge, "focusable", False):
            edge.focusable = True
            edge_options_changed = True

        seen_pairs.add(pair)
        cleaned_edges.append(edge)

    changed = len(cleaned_edges) != len(original_edges) or edge_options_changed

    if changed:
        st.session_state.builder_flow_state.edges = cleaned_edges

    return {
        "changed": changed,
        "invalid": removed_invalid,
        "duplicates": removed_duplicates,
        "self": removed_self,
        "removed_total": (
            removed_invalid
            + removed_duplicates
            + removed_self
        ),
    }


def selected_builder_edge():
    """Liefert die aktuell angeklickte Verbindung oder None."""
    init_builder_flow_state()

    selected_id = getattr(
        st.session_state.builder_flow_state,
        "selected_id",
        None
    )

    for edge in st.session_state.builder_flow_state.edges:
        if edge.id == selected_id or getattr(edge, "selected", False):
            return edge

    return None


def delete_builder_edge(edge_id: str):
    """Löscht gezielt eine Verbindung anhand ihrer ID."""
    init_builder_flow_state()

    st.session_state.builder_flow_state.edges = [
        edge
        for edge in st.session_state.builder_flow_state.edges
        if edge.id != edge_id
    ]

    try:
        st.session_state.builder_flow_state.selected_id = None
    except Exception:
        pass

    st.session_state.builder_last_validation = None


def validate_interactive_builder():
    """
    Prüft nicht nur Parameter, sondern den tatsächlich auf der Arbeitsfläche
    gezeichneten Signalweg.
    """
    init_builder_flow_state()
    config = st.session_state.builder_config
    errors = []
    warnings = []

    required_kinds = ["setpoint", "sum", "controller", "plant", "output", "feedback"]
    missing = [kind for kind in required_kinds if not builder_has_node(kind)]

    kind_names = {
        "setpoint": "Sollwert",
        "sum": "Vergleichsstelle",
        "controller": "Regler",
        "plant": "Strecke",
        "output": "Regelgröße / Ausgang",
        "feedback": "Rückführung",
    }

    for kind in missing:
        errors.append(f"Baustein fehlt: {kind_names[kind]}.")

    disturbance_position = config.get("disturbance_position", "Keine Störung")

    if disturbance_position != "Keine Störung":
        if not builder_has_node("disturbance"):
            errors.append("Der Störungsbaustein fehlt.")
        if not builder_has_node("dist_sum"):
            errors.append("Die Stör-Summierstelle fehlt.")

    # Nur Verbindungen prüfen, deren benötigte Knoten existieren.
    if not missing:
        base_connections = [
            ("sollwert", "vergleich", "Sollwert → Vergleichsstelle"),
            ("vergleich", "regler", "Vergleichsstelle → Regler"),
            ("ausgang", "rueckfuehrung", "Ausgang → Rückführung"),
            ("rueckfuehrung", "vergleich", "Rückführung → Vergleichsstelle"),
        ]

        if disturbance_position == "Keine Störung":
            base_connections.extend([
                ("regler", "strecke", "Regler → Strecke"),
                ("strecke", "ausgang", "Strecke → Ausgang"),
            ])

        elif disturbance_position == "Vor der Strecke":
            base_connections.extend([
                ("regler", "stoersumme", "Regler → Stör-Summierstelle"),
                ("stoerung", "stoersumme", "Störung → Stör-Summierstelle"),
                ("stoersumme", "strecke", "Stör-Summierstelle → Strecke"),
                ("strecke", "ausgang", "Strecke → Ausgang"),
            ])

        elif disturbance_position == "Am Ausgang":
            base_connections.extend([
                ("regler", "strecke", "Regler → Strecke"),
                ("strecke", "stoersumme", "Strecke → Stör-Summierstelle"),
                ("stoerung", "stoersumme", "Störung → Stör-Summierstelle"),
                ("stoersumme", "ausgang", "Stör-Summierstelle → Ausgang"),
            ])

        for source, target, description in base_connections:
            if not edge_exists(source, target):
                errors.append(f"Verbindung fehlt: {description}.")

    # Parameterprüfung
    if float(config.get("kp", 0.0)) < 0:
        errors.append("Kp darf nicht negativ sein.")

    if config.get("controller_type") in ["PI", "PID"] and float(config.get("ki", 0.0)) <= 0:
        warnings.append(
            "Ki ist 0. Der gewählte Regler besitzt dadurch praktisch keinen wirksamen I-Anteil."
        )

    if config.get("controller_type") == "PID" and float(config.get("kd", 0.0)) <= 0:
        warnings.append(
            "Kd ist 0. Der PID-Regler verhält sich dadurch praktisch wie ein PI-Regler."
        )

    if float(config.get("ks", 0.0)) <= 0:
        errors.append("Ks muss größer als 0 sein.")

    if config.get("plant_type") == "PT1" and float(config.get("ts", 0.0)) <= 0:
        errors.append("Ts muss größer als 0 sein.")

    if config.get("plant_type") == "PT2":
        if float(config.get("zeta", 0.0)) <= 0:
            errors.append("ζ muss größer als 0 sein.")
        if float(config.get("omega0", 0.0)) <= 0:
            errors.append("ω0 muss größer als 0 sein.")

    if float(config.get("dt", 0.0)) <= 0:
        errors.append("dt muss größer als 0 sein.")

    if float(config.get("t_end", 0.0)) <= 0:
        errors.append("Die Simulationsdauer muss größer als 0 sein.")

    if float(config.get("dt", 0.01)) >= float(config.get("t_end", 20.0)) / 20:
        warnings.append(
            "Die Schrittweite dt ist relativ groß. Für eine stabile Darstellung sollte sie deutlich kleiner sein."
        )

    return errors, warnings


def render_builder_step_header(step: int):
    steps = [
        "Aufgabe definieren",
        "Regler einbauen",
        "Strecke einbauen",
        "Rückführung & Störung",
        "Verbinden & prüfen",
    ]

    st.progress(step / len(steps))
    st.caption(
        f"Engineering-Schritt {step} von {len(steps)} · {steps[step - 1]}"
    )


def selected_builder_node():
    """Gibt den aktuell angeklickten Knoten zurück, falls einer ausgewählt ist."""
    init_builder_flow_state()
    selected_id = getattr(st.session_state.builder_flow_state, "selected_id", None)

    if not selected_id:
        return None

    for node in st.session_state.builder_flow_state.nodes:
        if node.id == selected_id:
            return node

    return None


def render_selected_node_editor():
    """
    Kompakter Eigenschaften-Dialog für den auf der Arbeitsfläche
    angeklickten Baustein.
    """
    node = selected_builder_node()

    if node is None:
        return

    kind = node.data.get("kind")
    config = st.session_state.builder_config

    readable = {
        "setpoint": "Sollwert",
        "sum": "Vergleichsstelle",
        "controller": "Regler",
        "plant": "Strecke",
        "output": "Regelgröße",
        "feedback": "Rückführung",
        "disturbance": "Störung",
        "dist_sum": "Stör-Summierstelle",
    }

    with st.expander(
        f"Ausgewählter Baustein: {readable.get(kind, node.id)}",
        expanded=True
    ):
        if kind == "controller":
            controller_type = st.selectbox(
                "Reglertyp",
                ["P", "PI", "PID"],
                index=["P", "PI", "PID"].index(config.get("controller_type", "PI")),
                key="ib_selected_controller"
            )

            kp = st.number_input(
                "Kp",
                min_value=0.0,
                value=float(config.get("kp", 2.0)),
                step=0.1,
                key="ib_selected_kp"
            )

            ki = 0.0
            kd = 0.0

            if controller_type in ["PI", "PID"]:
                ki = st.number_input(
                    "Ki",
                    min_value=0.0,
                    value=float(config.get("ki", 0.5)),
                    step=0.1,
                    key="ib_selected_ki"
                )

            if controller_type == "PID":
                kd = st.number_input(
                    "Kd",
                    min_value=0.0,
                    value=float(config.get("kd", 0.0)),
                    step=0.1,
                    key="ib_selected_kd"
                )

            if st.button("Regler-Eigenschaften übernehmen", key="ib_apply_controller"):
                config["controller_type"] = controller_type
                config["kp"] = kp
                config["ki"] = ki
                config["kd"] = kd
                st.session_state.builder_config = config
                sync_builder_nodes_from_config()
                st.rerun()

        elif kind == "plant":
            plant_type = st.selectbox(
                "Streckentyp",
                ["PT1", "PT2"],
                index=["PT1", "PT2"].index(config.get("plant_type", "PT1")),
                key="ib_selected_plant"
            )

            ks = st.number_input(
                "Ks",
                min_value=0.1,
                value=float(config.get("ks", 1.0)),
                step=0.1,
                key="ib_selected_ks"
            )

            ts = float(config.get("ts", 2.0))
            zeta = float(config.get("zeta", 0.7))
            omega0 = float(config.get("omega0", 2.0))

            if plant_type == "PT1":
                ts = st.number_input(
                    "Ts [s]",
                    min_value=0.1,
                    value=ts,
                    step=0.1,
                    key="ib_selected_ts"
                )
            else:
                zeta = st.number_input(
                    "ζ",
                    min_value=0.05,
                    value=zeta,
                    step=0.05,
                    key="ib_selected_zeta"
                )
                omega0 = st.number_input(
                    "ω0 [rad/s]",
                    min_value=0.1,
                    value=omega0,
                    step=0.1,
                    key="ib_selected_omega0"
                )

            if st.button("Strecken-Eigenschaften übernehmen", key="ib_apply_plant"):
                config["plant_type"] = plant_type
                config["ks"] = ks
                config["ts"] = ts
                config["zeta"] = zeta
                config["omega0"] = omega0
                st.session_state.builder_config = config
                sync_builder_nodes_from_config()
                st.rerun()

        elif kind == "disturbance":
            disturbance_time = st.number_input(
                "Störung ab [s]",
                min_value=0.0,
                value=float(config.get("disturbance_time", 8.0)),
                step=0.5,
                key="ib_selected_dist_time"
            )
            disturbance_value = st.number_input(
                "Störgröße d",
                value=float(config.get("disturbance_value", -0.3)),
                step=0.1,
                key="ib_selected_dist_value"
            )

            if st.button("Störung übernehmen", key="ib_apply_dist"):
                config["disturbance_time"] = disturbance_time
                config["disturbance_value"] = disturbance_value
                st.session_state.builder_config = config
                sync_builder_nodes_from_config()
                st.rerun()

        elif kind == "setpoint":
            setpoint_name = st.text_input(
                "Bezeichnung",
                value=config.get("setpoint_name", "Sollwert w(t)"),
                key="ib_selected_setpoint_name"
            )
            setpoint = st.number_input(
                "Sollwert",
                value=float(config.get("setpoint", 1.0)),
                step=0.1,
                key="ib_selected_setpoint"
            )

            if st.button("Sollwert übernehmen", key="ib_apply_setpoint"):
                config["setpoint_name"] = setpoint_name
                config["setpoint"] = setpoint
                st.session_state.builder_config = config
                sync_builder_nodes_from_config()
                st.rerun()

        elif kind == "output":
            output_name = st.text_input(
                "Bezeichnung der Regelgröße",
                value=config.get("output_name", "Regelgröße y(t)"),
                key="ib_selected_output_name"
            )

            if st.button("Regelgröße übernehmen", key="ib_apply_output"):
                config["output_name"] = output_name
                st.session_state.builder_config = config
                sync_builder_nodes_from_config()
                st.rerun()

        else:
            st.write(node.data.get("content", node.id))
            st.caption(
                "Dieser Baustein besitzt in der aktuellen Simulation keine zusätzlichen numerischen Parameter."
            )


def render_visual_builder():
    """
    Interaktiver, aber bewusst geführter Engineering-Baukasten.

    Links wird immer nur der aktuelle Arbeitsschritt gezeigt.
    Rechts ist die echte Arbeitsfläche: Bausteine verschieben, verbinden,
    anklicken sowie über Kontextmenüs löschen.
    """
    clear_pending_parameter_widgets()
    init_builder_flow_state()

    config = st.session_state.builder_config
    step = int(st.session_state.builder_step)

    st.title("Visueller Regelkreis-Builder")
    st.caption(
        "Baue den Regelkreis selbst auf. Die App führt dich dabei Schritt für Schritt durch den "
        "Engineering-Prozess, ohne die Arbeitsfläche mit allen Bausteinen gleichzeitig zu überladen."
    )

    render_builder_step_header(step)

    col_work = st.sidebar
    col_canvas = st.container()

    # --------------------------------------------------------
    # Linke Seite: genau ein Engineering-Schritt
    # --------------------------------------------------------
    with col_work:
        st.header("Bedienpanel")
        st.caption("Optionen links · interaktive Arbeitsfläche rechts")

        if "parameter_sync_notice" in st.session_state:
            st.success(st.session_state.pop("parameter_sync_notice"))

        if st.button(
            "Parameter aus Simulation übernehmen",
            width="stretch",
            key="ib_import_simulation",
            help="Übernimmt Reglertyp, Strecke, Sollwert, Störung und alle numerischen Parameter aus der Simulation.",
        ):
            apply_shared_parameters(shared_parameters_from_session(), "der Simulation")
            st.rerun()

        if step == 1:
            st.subheader("1. Regelaufgabe")

            st.write(
                "Definiere zuerst Führungs- und Regelgröße. Anschließend legst du die drei "
                "Grundbausteine auf die Arbeitsfläche."
            )

            config["setpoint_name"] = st.text_input(
                "Führungsgröße",
                value=config.get("setpoint_name", "Sollwert w(t)"),
                key="ib_setpoint_name"
            )

            config["output_name"] = st.text_input(
                "Regelgröße",
                value=config.get("output_name", "Regelgröße y(t)"),
                key="ib_output_name"
            )

            config["setpoint"] = st.number_input(
                "Sollwert w",
                value=float(config.get("setpoint", 1.0)),
                step=0.1,
                key="ib_setpoint"
            )

            sync_builder_nodes_from_config()

            st.markdown("**Grundbausteine**")

            if not builder_has_node("setpoint"):
                if st.button("Sollwert hinzufügen", width="stretch"):
                    add_builder_node("setpoint")
                    st.rerun()
            else:
                st.success("Sollwert liegt auf der Arbeitsfläche.")

            if not builder_has_node("sum"):
                if st.button("Vergleichsstelle hinzufügen", width="stretch"):
                    add_builder_node("sum")
                    st.rerun()
            else:
                st.success("Vergleichsstelle liegt auf der Arbeitsfläche.")

            if not builder_has_node("output"):
                if st.button("Ausgang / Regelgröße hinzufügen", width="stretch"):
                    add_builder_node("output")
                    st.rerun()
            else:
                st.success("Ausgang / Regelgröße liegt auf der Arbeitsfläche.")

            with st.expander("Warum diese drei Bausteine?", expanded=False):
                st.write(
                    "Der Sollwert beschreibt das gewünschte Verhalten. An der Vergleichsstelle wird "
                    "Soll- und Istwert verglichen. Die Regelgröße ist die Größe, die tatsächlich geregelt wird."
                )

        elif step == 2:
            st.subheader("2. Regler")

            st.write(
                "Wähle jetzt den Regler und parametriere ihn. Danach wird genau dieser Baustein "
                "auf die Arbeitsfläche gelegt."
            )

            config["controller_type"] = st.radio(
                "Reglertyp",
                ["P", "PI", "PID"],
                index=["P", "PI", "PID"].index(config.get("controller_type", "PI")),
                horizontal=True,
                key="ib_controller_type"
            )

            config["kp"] = st.number_input(
                "Kp",
                min_value=0.0,
                value=float(config.get("kp", 2.0)),
                step=0.1,
                key="ib_kp"
            )

            if config["controller_type"] in ["PI", "PID"]:
                config["ki"] = st.number_input(
                    "Ki",
                    min_value=0.0,
                    value=float(config.get("ki", 0.5)),
                    step=0.1,
                    key="ib_ki"
                )
            else:
                config["ki"] = 0.0

            if config["controller_type"] == "PID":
                config["kd"] = st.number_input(
                    "Kd",
                    min_value=0.0,
                    value=float(config.get("kd", 0.0)),
                    step=0.1,
                    key="ib_kd"
                )
            else:
                config["kd"] = 0.0

            sync_builder_nodes_from_config()

            if not builder_has_node("controller"):
                if st.button(
                    f"{config['controller_type']}-Regler hinzufügen",
                    type="primary",
                    width="stretch"
                ):
                    add_builder_node("controller")
                    st.rerun()
            else:
                st.success("Regler liegt auf der Arbeitsfläche.")
                st.caption(
                    "Du kannst den Regler anklicken, um seine Eigenschaften später erneut zu bearbeiten."
                )

            with st.expander("Auswahlhilfe", expanded=False):
                st.markdown(
                    """
                    **P:** einfache, schnelle Reaktion; bleibende Regelabweichung möglich.  
                    **PI:** beseitigt typischerweise die bleibende Regelabweichung.  
                    **PID:** ergänzt einen D-Anteil zur Beeinflussung schneller Änderungen und des Überschwingens.
                    """
                )

        elif step == 3:
            st.subheader("3. Strecke")

            st.write(
                "Modelliere nun das dynamische Verhalten der Anlage als PT1- oder PT2-Strecke."
            )

            config["plant_type"] = st.radio(
                "Streckentyp",
                ["PT1", "PT2"],
                index=["PT1", "PT2"].index(config.get("plant_type", "PT1")),
                horizontal=True,
                key="ib_plant_type"
            )

            config["ks"] = st.number_input(
                "Ks – Streckenverstärkung",
                min_value=0.1,
                value=float(config.get("ks", 1.0)),
                step=0.1,
                key="ib_ks"
            )

            if config["plant_type"] == "PT1":
                config["ts"] = st.number_input(
                    "Ts – Zeitkonstante [s]",
                    min_value=0.1,
                    value=float(config.get("ts", 2.0)),
                    step=0.1,
                    key="ib_ts"
                )
            else:
                config["zeta"] = st.number_input(
                    "ζ – Dämpfung",
                    min_value=0.05,
                    value=float(config.get("zeta", 0.7)),
                    step=0.05,
                    key="ib_zeta"
                )

                config["omega0"] = st.number_input(
                    "ω0 – Eigenkreisfrequenz [rad/s]",
                    min_value=0.1,
                    value=float(config.get("omega0", 2.0)),
                    step=0.1,
                    key="ib_omega0"
                )

            sync_builder_nodes_from_config()

            if not builder_has_node("plant"):
                if st.button(
                    f"{config['plant_type']}-Strecke hinzufügen",
                    type="primary",
                    width="stretch"
                ):
                    add_builder_node("plant")
                    st.rerun()
            else:
                st.success("Strecke liegt auf der Arbeitsfläche.")

            with st.expander("Auswahlhilfe", expanded=False):
                st.markdown(
                    """
                    **PT1:** typische träge Ausgleichsstrecke.  
                    **PT2:** System zweiter Ordnung; Dämpfung und Schwingverhalten können eine Rolle spielen.
                    """
                )

        elif step == 4:
            st.subheader("4. Rückführung und Störung")

            st.write(
                "Schließe den Regelkreis über die Rückführung. Eine Störung ist optional und wird nur "
                "eingeblendet, wenn du sie wirklich untersuchen möchtest."
            )

            if not builder_has_node("feedback"):
                if st.button("Rückführung hinzufügen", type="primary", width="stretch"):
                    add_builder_node("feedback")
                    st.rerun()
            else:
                st.success("Rückführung liegt auf der Arbeitsfläche.")

            st.divider()

            config["disturbance_position"] = st.selectbox(
                "Störung",
                ["Keine Störung", "Vor der Strecke", "Am Ausgang"],
                index=["Keine Störung", "Vor der Strecke", "Am Ausgang"].index(
                    config.get("disturbance_position", "Keine Störung")
                ),
                key="ib_disturbance_position"
            )

            if config["disturbance_position"] == "Keine Störung":
                # Bereits vorhandene Störbausteine bewusst entfernen.
                if builder_has_node("disturbance") or builder_has_node("dist_sum"):
                    if st.button("Störungsbausteine entfernen", width="stretch"):
                        remove_builder_node("disturbance")
                        remove_builder_node("dist_sum")
                        st.rerun()
                else:
                    st.caption("Es wird nur das Führungsverhalten untersucht.")

            else:
                config["disturbance_time"] = st.number_input(
                    "Störung ab [s]",
                    min_value=0.0,
                    value=float(config.get("disturbance_time", 8.0)),
                    step=0.5,
                    key="ib_disturbance_time"
                )

                config["disturbance_value"] = st.number_input(
                    "Störgröße d",
                    value=float(config.get("disturbance_value", -0.3)),
                    step=0.1,
                    key="ib_disturbance_value"
                )

                sync_builder_nodes_from_config()

                if not builder_has_node("disturbance"):
                    if st.button("Störungsbaustein hinzufügen", width="stretch"):
                        add_builder_node("disturbance")
                        st.rerun()
                else:
                    st.success("Störung liegt auf der Arbeitsfläche.")

                if not builder_has_node("dist_sum"):
                    if st.button("Stör-Summierstelle hinzufügen", width="stretch"):
                        add_builder_node("dist_sum")
                        st.rerun()
                else:
                    st.success("Stör-Summierstelle liegt auf der Arbeitsfläche.")

            with st.expander("Engineering-Hinweis", expanded=False):
                st.markdown(
                    """
                    **Vor der Strecke:** typischer Last- oder Prozesseinfluss auf den Streckeneingang.  
                    **Am Ausgang:** direkte Störung der Regelgröße.  
                    Die Störung wird über eine eigene Summierstelle in den Signalweg eingebracht.
                    """
                )

        else:
            st.subheader("5. Verbinden und prüfen")

            st.write(
                "Jetzt ist die Arbeitsfläche entscheidend: Ziehe die Verbindungen zwischen den Anschlusspunkten "
                "der Bausteine. Die App prüft anschließend genau den gezeichneten Signalweg."
            )

            config["t_end"] = st.number_input(
                "Simulationsdauer [s]",
                min_value=1.0,
                max_value=200.0,
                value=float(config.get("t_end", 20.0)),
                step=1.0,
                key="ib_t_end"
            )

            config["dt"] = st.number_input(
                "Schrittweite dt [s]",
                min_value=0.001,
                max_value=1.0,
                value=float(config.get("dt", 0.01)),
                step=0.001,
                format="%.3f",
                key="ib_dt"
            )

            st.info(
                "Verbindungen werden auf der Arbeitsfläche durch Ziehen vom Ausgang eines Bausteins "
                "zum Eingang des nächsten Bausteins erzeugt. Nicht unterstützte oder doppelte "
                "Verbindungen werden automatisch wieder entfernt."
            )

            if st.button("Regelkreis prüfen", type="primary", width="stretch"):
                errors, warnings = validate_interactive_builder()
                st.session_state.builder_last_validation = {
                    "errors": errors,
                    "warnings": warnings
                }

            result = st.session_state.builder_last_validation

            if result is not None:
                if result["errors"]:
                    st.error("Der gezeichnete Regelkreis ist noch nicht vollständig.")
                    for item in result["errors"]:
                        st.write(f"- {item}")
                else:
                    st.success("Der gezeichnete Regelkreis ist vollständig und simulationsfähig.")

                for item in result["warnings"]:
                    st.warning(item)

                if not result["errors"]:
                    if st.button(
                        "Gezeichneten Regelkreis in Simulation übernehmen",
                        type="primary",
                        width="stretch"
                    ):
                        apply_shared_parameters(
                            config, "dem visuellen Regelkreis-Builder"
                        )
                        st.session_state.active_view = "simulation"
                        st.rerun()

            with st.expander("Soll-Verbindungen anzeigen", expanded=False):
                if config.get("disturbance_position") == "Keine Störung":
                    st.markdown(
                        """
                        1. Sollwert → Vergleichsstelle  
                        2. Vergleichsstelle → Regler  
                        3. Regler → Strecke  
                        4. Strecke → Regelgröße  
                        5. Regelgröße → Rückführung  
                        6. Rückführung → Vergleichsstelle
                        """
                    )
                elif config.get("disturbance_position") == "Vor der Strecke":
                    st.markdown(
                        """
                        1. Sollwert → Vergleichsstelle  
                        2. Vergleichsstelle → Regler  
                        3. Regler → Stör-Summierstelle  
                        4. Störung → Stör-Summierstelle  
                        5. Stör-Summierstelle → Strecke  
                        6. Strecke → Regelgröße  
                        7. Regelgröße → Rückführung  
                        8. Rückführung → Vergleichsstelle
                        """
                    )
                else:
                    st.markdown(
                        """
                        1. Sollwert → Vergleichsstelle  
                        2. Vergleichsstelle → Regler  
                        3. Regler → Strecke  
                        4. Strecke → Stör-Summierstelle  
                        5. Störung → Stör-Summierstelle  
                        6. Stör-Summierstelle → Regelgröße  
                        7. Regelgröße → Rückführung  
                        8. Rückführung → Vergleichsstelle
                        """
                    )

        st.session_state.builder_config = config
        sync_builder_nodes_from_config()

        st.divider()
        if st.button(
            "Parameter an alle Bereiche übertragen",
            type="primary",
            width="stretch",
            key="ib_sync_all",
            help="Übernimmt den aktuellen Builder-Stand in Simulation und physikalischen Wirkplan-Builder.",
        ):
            apply_shared_parameters(config, "dem visuellen Regelkreis-Builder")
            st.rerun()

        if st.button(
            "Parameter auf Standard zurücksetzen",
            width="stretch",
            key="ib_reset_parameters",
            help="Setzt die gemeinsam genutzten Regler-, Strecken- und Simulationsparameter zurück; die Bausteine bleiben erhalten.",
        ):
            reset_shared_parameters()
            st.rerun()

        # Kompakter Editor erscheint nur, wenn auf der Arbeitsfläche
        # wirklich ein Baustein angeklickt wurde.
        render_selected_node_editor()

        # ----------------------------------------------------
        # Schritt-Navigation
        # ----------------------------------------------------
        st.divider()
        back_col, next_col = st.columns(2)

        with back_col:
            if step > 1:
                if st.button("Zurück", width="stretch", key="ib_back"):
                    st.session_state.builder_step = step - 1
                    st.session_state.builder_last_validation = None
                    st.rerun()

        with next_col:
            if step < 5:
                step_ready = True

                if step == 1:
                    step_ready = (
                        builder_has_node("setpoint")
                        and builder_has_node("sum")
                        and builder_has_node("output")
                    )
                elif step == 2:
                    step_ready = builder_has_node("controller")
                elif step == 3:
                    step_ready = builder_has_node("plant")
                elif step == 4:
                    step_ready = builder_has_node("feedback")
                    if config.get("disturbance_position") != "Keine Störung":
                        step_ready = (
                            step_ready
                            and builder_has_node("disturbance")
                            and builder_has_node("dist_sum")
                        )

                if st.button(
                    "Weiter",
                    type="primary",
                    width="stretch",
                    key="ib_next",
                    disabled=not step_ready
                ):
                    st.session_state.builder_step = step + 1
                    st.session_state.builder_last_validation = None
                    st.rerun()

        if st.button("Arbeitsfläche leeren / neu beginnen", width="stretch", key="ib_reset"):
            reset_interactive_builder()
            st.rerun()

    # --------------------------------------------------------
    # Rechte Seite: echte interaktive Arbeitsfläche
    # --------------------------------------------------------
    with col_canvas:
        st.subheader("Arbeitsfläche")

        st.caption(
            "Bausteine verschieben · zulässige Anschlüsse verbinden · Verbindung anklicken und mit ✂ löschen · "
            "Rechtsklick auf Baustein oder Verbindung für das Kontextmenü."
        )

        st.caption(
            "Der Ausgang / die Regelgröße befindet sich rechts am Ende der Hauptkette. "
            "Mit dem Fit-View-Steuerelement kannst du jederzeit alle Bausteine wieder ins Bild holen."
        )

        st.session_state.builder_flow_state = streamlit_flow(
            "interactive_engineering_builder",
            st.session_state.builder_flow_state,
            fit_view=True,
            height=650,
            show_minimap=False,
            show_controls=True,
            hide_watermark=True,
            allow_new_edges=True,
            enable_node_menu=True,
            enable_edge_menu=True,
            enable_pane_menu=False,
            get_node_on_click=True,
            get_edge_on_click=True,
            min_zoom=0.2
        )

        # ----------------------------------------------------
        # Verbindungen fachlich begrenzen
        # ----------------------------------------------------
        cleanup = sanitize_builder_edges()

        if cleanup["changed"]:
            parts = []

            if cleanup["invalid"]:
                parts.append(
                    f"{cleanup['invalid']} fachlich nicht unterstützte"
                )

            if cleanup["duplicates"]:
                parts.append(
                    f"{cleanup['duplicates']} doppelte"
                )

            if cleanup["self"]:
                parts.append(
                    f"{cleanup['self']} selbstbezügliche"
                )

            st.session_state.builder_connection_notice = (
                ", ".join(parts)
                + " Verbindung(en) wurden automatisch entfernt."
            )

            # Der bereinigte Python-State wird damit sofort wieder in die
            # Flow-Komponente synchronisiert.
            st.rerun()

        if "builder_connection_notice" in st.session_state:
            st.warning(st.session_state.builder_connection_notice)
            del st.session_state.builder_connection_notice

        # ----------------------------------------------------
        # Verbindung gezielt löschen
        # ----------------------------------------------------
        selected_edge = selected_builder_edge()

        tool_col1, tool_col2 = st.columns([1.3, 2.7])

        with tool_col1:
            if selected_edge is not None:
                if st.button(
                    "✂ Verbindung löschen",
                    type="secondary",
                    width="stretch",
                    key="ib_delete_selected_edge"
                ):
                    delete_builder_edge(selected_edge.id)
                    st.rerun()
            else:
                st.button(
                    "✂ Verbindung löschen",
                    width="stretch",
                    disabled=True,
                    key="ib_delete_selected_edge_disabled"
                )

        with tool_col2:
            if selected_edge is not None:
                st.info(
                    f"Ausgewählt: {selected_edge.source} → {selected_edge.target}"
                )
            else:
                st.caption(
                    "Zum Löschen zuerst eine Verbindung anklicken. Alternativ: "
                    "Rechtsklick auf eine Verbindung und im Kontextmenü löschen."
                )

        # Änderungen aus der UI (Verschieben, neue Kanten, Löschen)
        # liegen jetzt wieder in builder_flow_state vor.
        node_count = len(st.session_state.builder_flow_state.nodes)
        edge_count = len(st.session_state.builder_flow_state.edges)

        status1, status2, status3 = st.columns(3)
        status1.metric("Bausteine", node_count)
        status2.metric("Verbindungen", edge_count)
        status3.metric("Schritt", f"{step}/5")

        if step < 5:
            hints = {
                1: "Lege zuerst Sollwert, Vergleichsstelle und Regelgröße auf die Fläche.",
                2: "Füge den Regler hinzu. Du kannst vorhandene Bausteine bereits passend anordnen.",
                3: "Füge die Strecke hinzu. Die Signalrichtung soll später von links nach rechts laufen.",
                4: "Ergänze Rückführung und optional die Störung.",
            }
            st.info(hints[step])
        else:
            st.success(
                "Die Arbeitsfläche ist jetzt der maßgebende Aufbau. "
                "Verbinde die Bausteine und prüfe anschließend den Regelkreis links."
            )


# ------------------------------------------------------------
# Physikalischer Wirkplan-Builder
# ------------------------------------------------------------

def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def ensure_real_process_defaults(config: dict):
    """Ergänzt neue Felder auch in bereits laufenden Streamlit-Sitzungen."""
    defaults = {
        "anlagenart": "RLT / Lüftung",
        "regelungsvariante": "Zulufttemperatur",
        "stellglied_typ": "2-Wege-Regelventil",
        "regelstrategie": "Automatische Empfehlung",
        "reale_daten_aktiv": True,
        "eingabetiefe": "Einfach",
        "temp_betriebsart": "Heizen",
        "temp_medium": "Wasser",
        "temp_volumen_m3": 1.0,
        "temp_heizleistung_kw": 12.0,
        "temp_umgebung_c": 15.0,
        "temp_soll_c": 55.0,
        "temp_waermeverlust_w_k": 180.0,
        "temp_wirkungsgrad": 0.95,
        "temp_dichte_kg_m3": 998.0,
        "temp_cp_kj_kgk": 4.18,
        "motor_leistung_kw": 5.5,
        "motor_nenndrehzahl_rpm": 1450.0,
        "motor_soll_rpm": 1200.0,
        "motor_hochlaufzeit_s": 5.0,
        "motor_leitung_m": 50.0,
        "motor_querschnitt_mm2": 2.5,
        "motor_spannung_v": 400.0,
        "motor_wirkungsgrad": 0.88,
        "motor_leistungsfaktor": 0.82,
        "motor_traegheit_kgm2": 0.18,
        "motor_lastmoment_nm": 20.0,
        "tank_form": "Zylindrisch",
        "tank_volumen_m3": 2.5,
        "tank_hoehe_m": 2.0,
        "tank_zulauf_m3h": 8.0,
        "tank_abfluss_m3h": 3.0,
        "tank_soll_m": 1.4,
        "druck_volumen_m3": 1.0,
        "druck_max_bar": 10.0,
        "druck_soll_bar": 6.0,
        "druck_foerderstrom_m3h": 60.0,
        "druck_verbrauch_m3h": 20.0,
        "druck_zeitkonstante_s": 3.0,
        "flow_max_m3h": 100.0,
        "flow_soll_m3h": 60.0,
        "flow_rohrlaenge_m": 20.0,
        "flow_durchmesser_mm": 80.0,
        "flow_ventilzeit_s": 0.8,
        "flow_druckverlust_bar": 1.5,
        "pos_masse_kg": 25.0,
        "pos_feder_n_m": 1200.0,
        "pos_daempfung_ns_m": 180.0,
        "pos_stellkraft_n": 1500.0,
        "pos_hub_mm": 500.0,
        "pos_soll_mm": 250.0,
        "generic_plant_type": "PT1",
        "generic_ks": 1.0,
        "generic_ts": 10.0,
        "generic_zeta": 0.7,
        "generic_omega0": 1.0,
        "generic_setpoint": 1.0,
        "generic_unit": "Prozesseinheit",
        "generic_t_end": 60.0,
        "sensor": "Temperaturfühler",
        "messglied": "Messumformer",
        "stellglied": "Leistungssteller / Heizung",
        "sollwertgeber": "Sollwertvorgabe",
        "messbereich_min": 0.0,
        "messbereich_max": 100.0,
        "sensor_zeitkonstante_s": 0.2,
        "messrauschen": 0.0,
        "totzeit_s": 0.0,
        "stellgroesse_min": 0.0,
        "stellgroesse_max": 100.0,
        "stellrate_max": 100.0,
        "auslegung": "Automatisch",
        "man_controller_type": "PI",
        "man_plant_type": "PT1",
        "man_kp": 2.0,
        "man_ki": 0.4,
        "man_kd": 0.0,
        "man_ks": 1.0,
        "man_ts": 3.0,
        "man_zeta": 0.7,
        "man_omega0": 2.0,
        "man_setpoint": 1.0,
        "man_t_end": 25.0,
        "man_dt": 0.01,
        "stoerort": "Vor der Strecke",
        "stoerzeit_s": 10.0,
        "stoerwert": -0.3,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    return config


def validate_wirkplan_config(config: dict):
    """Prüft prozessübergreifende Grenzen, bevor Daten in die Simulation gelangen."""
    errors = []
    if float(config.get("stellgroesse_min", 0.0)) >= float(config.get("stellgroesse_max", 100.0)):
        errors.append("Die maximale Stellgröße muss größer als die minimale Stellgröße sein.")
    if float(config.get("messbereich_min", 0.0)) >= float(config.get("messbereich_max", 100.0)):
        errors.append("Das Messbereichsmaximum muss größer als das Messbereichsminimum sein.")
    if config.get("auslegung") == "Manuell":
        if float(config.get("man_dt", 0.01)) >= float(config.get("man_t_end", 25.0)):
            errors.append("Der Zeitschritt dt muss kleiner als die Simulationsdauer sein.")
        if config.get("man_controller_type") in ["PI", "PID"] and float(config.get("man_ki", 0.0)) <= 0:
            errors.append("Ein PI-/PID-Regler benötigt Ki > 0.")
        if config.get("man_controller_type") == "PID" and float(config.get("man_kd", 0.0)) <= 0:
            errors.append("Ein PID-Regler benötigt Kd > 0.")
    return errors


def calculate_real_process_data(config: dict):
    """Berechnet aus realen Anlagendaten ein nachvollziehbares PT1-Ersatzmodell."""
    prozessart = config["prozessart"]
    result = {
        "active": bool(config.get("reale_daten_aktiv", True)),
        "supported": prozessart in {
            "Temperaturregelung", "Drehzahlregelung", "Füllstandsregelung",
            "Druckregelung", "Durchflussregelung", "Position / Mechanik",
            "Generische Prozessstrecke"
        },
        "metrics": [],
        "warnings": [],
        "node_details": {},
        "begruendung": [],
    }

    if not result["active"] or not result["supported"]:
        return result

    if prozessart == "Temperaturregelung":
        betriebsart = config.get("temp_betriebsart", "Heizen")
        medium = config.get("temp_medium", "Wasser")
        stoffwerte = {
            "Wasser": (998.0, 4.18),
            "Luft": (1.204, 1.005),
        }
        if medium in stoffwerte:
            dichte, cp = stoffwerte[medium]
        else:
            dichte = max(float(config.get("temp_dichte_kg_m3", 998.0)), 0.001)
            cp = max(float(config.get("temp_cp_kj_kgk", 4.18)), 0.001)

        volumen = max(float(config.get("temp_volumen_m3", 1.0)), 0.001)
        heizleistung_kw = max(float(config.get("temp_heizleistung_kw", 12.0)), 0.001)
        wirkungsgrad = _clamp(config.get("temp_wirkungsgrad", 0.95), 0.01, 1.0)
        waermeverlust = max(float(config.get("temp_waermeverlust_w_k", 180.0)), 0.1)
        umgebung = float(config.get("temp_umgebung_c", 15.0))
        solltemperatur = float(config.get("temp_soll_c", 55.0))

        masse = volumen * dichte
        waermekapazitaet_j_k = masse * cp * 1000.0
        wirksame_heizleistung_w = heizleistung_kw * 1000.0 * wirkungsgrad
        zeitkonstante_s = waermekapazitaet_j_k / waermeverlust
        max_delta_t = wirksame_heizleistung_w / waermeverlust
        soll_delta_t = max(0.1, abs(solltemperatur - umgebung))
        anfangssteigung_k_min = wirksame_heizleistung_w / waermekapazitaet_j_k * 60.0

        if soll_delta_t > max_delta_t:
            result["warnings"].append(
                f"Die gewünschte Temperatur ist mit der eingetragenen "
                f"{'Heiz' if betriebsart == 'Heizen' else 'Kühl'}leistung und dem "
                "Wärmeübergang im stationären Zustand nicht erreichbar."
            )
        if medium == "Luft":
            result["warnings"].append(
                "Bei Räumen speichert nicht nur die Luft Wärme. Für ein genaueres Modell "
                "müssen Wände, Einrichtung und Gebäudemasse als äquivalente Masse ergänzt werden."
            )

        result.update({
            "ks": max_delta_t / 100.0,
            "ts": _clamp(zeitkonstante_s, 0.1, 500000.0),
            "setpoint": soll_delta_t,
            "t_end": _clamp(5.0 * zeitkonstante_s, 30.0, 1000000.0),
            "disturbance_value": -10.0,
            "input_unit": f"% {'Heiz' if betriebsart == 'Heizen' else 'Kühl'}leistung",
            "output_unit": "K Temperaturänderung",
            "model_note": (
                f"Die Simulation regelt den Betrag der Temperaturänderung ΔT beim {betriebsart.lower()} gegenüber der "
                f"Umgebungstemperatur {umgebung:.1f} °C."
            ),
        })
        result["metrics"] = [
            ("Masse", f"{masse:.1f} kg"),
            ("Wärmekapazität", f"{waermekapazitaet_j_k / 1e6:.2f} MJ/K"),
            ("Zeitkonstante", f"{zeitkonstante_s / 60.0:.1f} min"),
            ("Anfangssteigung", f"{anfangssteigung_k_min:.3f} K/min"),
            ("max. ΔT", f"{max_delta_t:.1f} K"),
            ("Sollwert ΔT", f"{soll_delta_t:.1f} K"),
        ]
        result["node_details"] = {
            "stellgroesse": f"{betriebsart} {heizleistung_kw:.1f} kW",
            "prozessglied": f"{medium}, η = {wirkungsgrad * 100:.0f} %",
            "speicher": f"{volumen:.2f} m³ / {masse:.0f} kg",
            "regelgroesse": f"{solltemperatur:.1f} °C (ΔT {soll_delta_t:.1f} K)",
        }
        result["begruendung"].append(
            "Die thermische Zeitkonstante folgt aus Wärmekapazität geteilt durch Wärmeverlustkoeffizient."
        )

    elif prozessart == "Drehzahlregelung":
        leistung_kw = max(float(config.get("motor_leistung_kw", 5.5)), 0.01)
        nenndrehzahl = max(float(config.get("motor_nenndrehzahl_rpm", 1450.0)), 1.0)
        solldrehzahl = _clamp(config.get("motor_soll_rpm", 1200.0), 0.0, nenndrehzahl)
        hochlaufzeit = max(float(config.get("motor_hochlaufzeit_s", 5.0)), 0.1)
        leitung = max(float(config.get("motor_leitung_m", 50.0)), 0.0)
        querschnitt = max(float(config.get("motor_querschnitt_mm2", 2.5)), 0.1)
        spannung = max(float(config.get("motor_spannung_v", 400.0)), 1.0)
        wirkungsgrad = _clamp(config.get("motor_wirkungsgrad", 0.88), 0.1, 1.0)
        cos_phi = _clamp(config.get("motor_leistungsfaktor", 0.82), 0.1, 1.0)
        traegheit = max(float(config.get("motor_traegheit_kgm2", 0.18)), 0.0001)
        lastmoment = max(float(config.get("motor_lastmoment_nm", 20.0)), 0.0)

        nennmoment = 9550.0 * leistung_kw / nenndrehzahl
        nennstrom = leistung_kw * 1000.0 / (
            np.sqrt(3.0) * spannung * wirkungsgrad * cos_phi
        )
        spannungsfall = np.sqrt(3.0) * nennstrom * 0.0178 * leitung / querschnitt
        spannungsfall_prozent = spannungsfall / spannung * 100.0
        motorspannung = max(spannung - spannungsfall, 0.0)
        momentfaktor = (motorspannung / spannung) ** 2
        verfuegbares_moment = nennmoment * momentfaktor
        beschleunigungsmoment = verfuegbares_moment - lastmoment
        omega_n = 2.0 * np.pi * nenndrehzahl / 60.0

        if beschleunigungsmoment > 0:
            mechanische_hochlaufzeit = traegheit * omega_n / beschleunigungsmoment
        else:
            mechanische_hochlaufzeit = hochlaufzeit
            result["warnings"].append(
                "Das berechnete Motormoment ist nicht größer als das Lastmoment. "
                "Ein sicherer Hochlauf ist mit diesen Angaben nicht nachgewiesen."
            )
        if spannungsfall_prozent > 3.0:
            result["warnings"].append(
                f"Der überschlägige Spannungsfall beträgt {spannungsfall_prozent:.1f} %. "
                "Leitung, Verlegeart, Schutzorgan und Anlaufstrom müssen genauer geprüft werden."
            )

        pt1_zeit = max(hochlaufzeit, mechanische_hochlaufzeit) / 3.0
        result.update({
            "ks": nenndrehzahl / 100.0,
            "ts": _clamp(pt1_zeit, 0.1, 500000.0),
            "setpoint": solldrehzahl,
            "t_end": max(20.0, 6.0 * pt1_zeit),
            "disturbance_value": -10.0,
            "input_unit": "% Ansteuerung",
            "output_unit": "1/min",
            "model_note": (
                "Der Spannungsfall ist eine überschlägige Drehstromberechnung. "
                "Bei Frequenzumrichterbetrieb sind zusätzlich Herstellerangaben zu Motorkabel, "
                "Filter und EMV zu beachten."
            ),
        })
        result["metrics"] = [
            ("Nennmoment", f"{nennmoment:.1f} Nm"),
            ("Nennstrom ca.", f"{nennstrom:.1f} A"),
            ("Spannungsfall", f"{spannungsfall:.1f} V / {spannungsfall_prozent:.1f} %"),
            ("Moment an Leitung", f"{verfuegbares_moment:.1f} Nm"),
            ("mechan. Hochlauf", f"{mechanische_hochlaufzeit:.2f} s"),
            ("PT1-Zeitkonstante", f"{pt1_zeit:.2f} s"),
        ]
        result["node_details"] = {
            "stellgroesse": f"0–100 % / {spannung:.0f} V",
            "prozessglied": f"Motor {leistung_kw:.1f} kW, Leitung {leitung:.0f} m",
            "speicher": f"J = {traegheit:.3f} kg·m²",
            "regelgroesse": f"Soll {solldrehzahl:.0f} 1/min",
        }
        result["begruendung"].append(
            "Das Motormoment wird aus Leistung und Nenndrehzahl berechnet; die längere aus vorgegebener und mechanischer Hochlaufzeit bestimmt das PT1-Ersatzmodell."
        )

    elif prozessart == "Füllstandsregelung":
        volumen = max(float(config.get("tank_volumen_m3", 2.5)), 0.001)
        hoehe = max(float(config.get("tank_hoehe_m", 2.0)), 0.01)
        zulauf = max(float(config.get("tank_zulauf_m3h", 8.0)), 0.001)
        abfluss = max(float(config.get("tank_abfluss_m3h", 3.0)), 0.0)
        soll = _clamp(config.get("tank_soll_m", 1.4), 0.0, hoehe)
        querschnitt = volumen / hoehe
        netto_zulauf = zulauf - abfluss

        if netto_zulauf > 0:
            vollfuellzeit_s = volumen / netto_zulauf * 3600.0
            sollzeit_s = querschnitt * soll / netto_zulauf * 3600.0
        else:
            vollfuellzeit_s = volumen / zulauf * 3600.0
            sollzeit_s = querschnitt * soll / zulauf * 3600.0
            result["warnings"].append(
                "Der maximale Zulauf ist nicht größer als der Abfluss. Der Sollfüllstand kann bei konstantem Abfluss nicht erreicht werden."
            )

        pt1_zeit = max(vollfuellzeit_s / 3.0, 0.1)
        if config.get("tank_form") == "Zylindrisch":
            durchmesser = 2.0 * np.sqrt(querschnitt / np.pi)
            geometrie = f"Ø {durchmesser:.2f} m"
        else:
            durchmesser = None
            geometrie = f"A = {querschnitt:.2f} m²"

        result.update({
            "ks": hoehe / 100.0,
            "ts": _clamp(pt1_zeit, 0.1, 500000.0),
            "setpoint": soll,
            "t_end": _clamp(max(60.0, 5.0 * pt1_zeit), 60.0, 1000000.0),
            "disturbance_value": -10.0 if abfluss > 0 else -5.0,
            "input_unit": "% Zulauf",
            "output_unit": "m Füllstand",
            "model_note": (
                "Ein Behälter ist physikalisch eine integrierende Strecke. Für das vorhandene "
                "Regelkreis-Labor wird daraus zunächst ein PT1-Ersatzmodell gebildet."
            ),
        })
        result["metrics"] = [
            ("Querschnitt", f"{querschnitt:.3f} m²"),
            ("Geometrie", geometrie),
            ("Nettozulauf", f"{netto_zulauf:.2f} m³/h"),
            ("Zeit bis Soll", f"{sollzeit_s / 60.0:.1f} min"),
            ("Zeit bis voll", f"{vollfuellzeit_s / 60.0:.1f} min"),
            ("PT1-Ersatzzeit", f"{pt1_zeit / 60.0:.1f} min"),
        ]
        result["node_details"] = {
            "stellgroesse": f"Zulauf bis {zulauf:.1f} m³/h",
            "prozessglied": f"Pumpe / Ventil, Abfluss {abfluss:.1f} m³/h",
            "speicher": f"{volumen:.2f} m³ / {hoehe:.2f} m",
            "regelgroesse": f"Soll {soll:.2f} m",
        }
        result["begruendung"].append(
            "Querschnitt, Nettozulauf und Zielhöhe bestimmen die reale Füllzeit; daraus wird ein PT1-Ersatzmodell für die vorhandene Simulation abgeleitet."
        )

    elif prozessart == "Druckregelung":
        volumen = max(float(config.get("druck_volumen_m3", 1.0)), 0.001)
        p_max = max(float(config.get("druck_max_bar", 10.0)), 0.01)
        p_soll = _clamp(config.get("druck_soll_bar", 6.0), 0.0, p_max)
        foerderstrom = max(float(config.get("druck_foerderstrom_m3h", 60.0)), 0.001)
        verbrauch = max(float(config.get("druck_verbrauch_m3h", 20.0)), 0.0)
        vorgabe_ts = max(float(config.get("druck_zeitkonstante_s", 3.0)), 0.05)
        netto = foerderstrom - verbrauch
        fuellzeit = volumen / max(abs(netto), 0.001) * 3600.0
        ts = max(vorgabe_ts, fuellzeit / 3.0)
        if netto <= 0:
            result["warnings"].append(
                "Der Förderstrom ist nicht größer als der Verbrauch; der Solldruck ist so nicht dauerhaft erreichbar."
            )
        result.update({
            "plant_type": "PT1", "ks": p_max / 100.0, "ts": ts,
            "setpoint": p_soll, "t_end": max(20.0, 6.0 * ts),
            "disturbance_value": -max(0.1, 0.1 * p_soll),
            "input_unit": "% Verdichterleistung", "output_unit": "bar",
            "model_note": "Das Druckmodell nutzt Behältervolumen, Förderstrom, Verbrauch und eine minimale Anlagenzeitkonstante als PT1-Näherung.",
        })
        result["metrics"] = [
            ("Behältervolumen", f"{volumen:.3f} m³"), ("Netto-Förderstrom", f"{netto:.2f} m³/h"),
            ("Füllzeit", f"{fuellzeit:.1f} s"), ("PT1-Zeitkonstante", f"{ts:.2f} s"),
            ("Solldruck", f"{p_soll:.2f} bar"), ("Maximaldruck", f"{p_max:.2f} bar"),
        ]
        result["node_details"] = {
            "stellgroesse": f"0–100 %, {foerderstrom:.1f} m³/h", "prozessglied": "Verdichter / Ventil",
            "speicher": f"{volumen:.2f} m³ Druckspeicher", "regelgroesse": f"Soll {p_soll:.2f} bar",
        }
        result["begruendung"].append("Speichervolumen und Netto-Förderstrom bestimmen die Druckdynamik des PT1-Ersatzmodells.")

    elif prozessart == "Durchflussregelung":
        q_max = max(float(config.get("flow_max_m3h", 100.0)), 0.001)
        q_soll = _clamp(config.get("flow_soll_m3h", 60.0), 0.0, q_max)
        laenge = max(float(config.get("flow_rohrlaenge_m", 20.0)), 0.0)
        durchmesser = max(float(config.get("flow_durchmesser_mm", 80.0)), 1.0)
        ventilzeit = max(float(config.get("flow_ventilzeit_s", 0.8)), 0.01)
        druckverlust = max(float(config.get("flow_druckverlust_bar", 1.5)), 0.0)
        rohrvolumen = np.pi * (durchmesser / 2000.0) ** 2 * laenge
        transportzeit = rohrvolumen / q_max * 3600.0
        ts = max(ventilzeit, transportzeit)
        result.update({
            "plant_type": "PT1", "ks": q_max / 100.0, "ts": ts,
            "setpoint": q_soll, "t_end": max(10.0, 8.0 * ts),
            "disturbance_value": -max(0.1, 0.1 * q_soll),
            "input_unit": "% Ventilöffnung", "output_unit": "m³/h",
            "model_note": "Ventildynamik und Transportzeit im Rohr werden zu einer robusten PT1-Näherung zusammengefasst.",
        })
        result["metrics"] = [
            ("Rohrvolumen", f"{rohrvolumen:.3f} m³"), ("Transportzeit", f"{transportzeit:.2f} s"),
            ("Ventilzeit", f"{ventilzeit:.2f} s"), ("PT1-Zeitkonstante", f"{ts:.2f} s"),
            ("Sollfluss", f"{q_soll:.2f} m³/h"), ("Druckverlust", f"{druckverlust:.2f} bar"),
        ]
        result["node_details"] = {
            "stellgroesse": "Ventil 0–100 %", "prozessglied": f"Rohr DN {durchmesser:.0f}, {laenge:.1f} m",
            "speicher": f"Rohrvolumen {rohrvolumen:.3f} m³", "regelgroesse": f"Soll {q_soll:.1f} m³/h",
        }
        result["begruendung"].append("Ventilzeit und Rohrvolumen bestimmen die Durchflussdynamik.")

    elif prozessart == "Position / Mechanik":
        masse = max(float(config.get("pos_masse_kg", 25.0)), 0.001)
        feder = max(float(config.get("pos_feder_n_m", 1200.0)), 0.001)
        daempfung = max(float(config.get("pos_daempfung_ns_m", 180.0)), 0.0)
        kraft = max(float(config.get("pos_stellkraft_n", 1500.0)), 0.001)
        hub = max(float(config.get("pos_hub_mm", 500.0)), 0.001)
        soll = _clamp(config.get("pos_soll_mm", 250.0), 0.0, hub)
        omega0 = np.sqrt(feder / masse)
        zeta = daempfung / (2.0 * np.sqrt(feder * masse))
        ts_equiv = 1.0 / max(omega0, 0.001)
        result.update({
            "plant_type": "PT2", "ks": hub / 100.0, "ts": ts_equiv,
            "zeta": _clamp(zeta, 0.05, 3.0), "omega0": omega0,
            "setpoint": soll, "t_end": max(10.0, 10.0 / max(omega0, 0.001)),
            "disturbance_value": -max(0.1, 0.05 * soll),
            "input_unit": "% Stellkraft", "output_unit": "mm",
            "model_note": "Masse, Feder und Dämpfung bilden ein physikalisches PT2-Modell.",
        })
        result["metrics"] = [
            ("Masse", f"{masse:.2f} kg"), ("Stellkraft", f"{kraft:.1f} N"),
            ("Eigenkreisfrequenz", f"{omega0:.3f} rad/s"), ("Dämpfungsgrad", f"{zeta:.3f}"),
            ("Hub", f"{hub:.1f} mm"), ("Sollposition", f"{soll:.1f} mm"),
        ]
        result["node_details"] = {
            "stellgroesse": f"Stellkraft {kraft:.0f} N", "prozessglied": "Antrieb / Mechanik",
            "speicher": f"m={masse:.1f} kg, k={feder:.0f} N/m", "regelgroesse": f"Soll {soll:.1f} mm",
        }
        result["begruendung"].append("Masse, Federsteifigkeit und Dämpfung bestimmen Eigenfrequenz und Dämpfungsgrad des PT2-Modells.")

    elif prozessart == "Generische Prozessstrecke":
        plant_type = config.get("generic_plant_type", "PT1")
        ks = max(float(config.get("generic_ks", 1.0)), 0.000001)
        ts = max(float(config.get("generic_ts", 10.0)), 0.000001)
        zeta = max(float(config.get("generic_zeta", 0.7)), 0.01)
        omega0 = max(float(config.get("generic_omega0", 1.0)), 0.000001)
        setpoint = float(config.get("generic_setpoint", 1.0))
        unit = config.get("generic_unit", "Prozesseinheit")
        t_end = max(float(config.get("generic_t_end", 60.0)), 0.1)
        result.update({
            "plant_type": plant_type, "ks": ks, "ts": ts,
            "zeta": zeta, "omega0": omega0, "setpoint": setpoint,
            "t_end": t_end, "disturbance_value": -0.1 * setpoint,
            "input_unit": "% Stellgröße", "output_unit": unit,
            "model_note": (
                "Diese praxisnahe Variante nutzt eine frei parametrierbare Ersatzstrecke. "
                "Ks, Ts und bei PT2 zusätzlich ζ und ω0 können an Messdaten angepasst werden."
            ),
        })
        result["metrics"] = [
            ("Streckentyp", plant_type), ("Streckenverstärkung Ks", f"{ks:.4g}"),
            ("Zeitkonstante Ts", f"{ts:.3f} s"), ("Sollwert", f"{setpoint:.4g} {unit}"),
            ("Dämpfung ζ", f"{zeta:.3f}"), ("Eigenkreisfrequenz ω0", f"{omega0:.3f} rad/s"),
        ]
        result["node_details"] = {
            "stellgroesse": config.get("stellglied_typ", "Stellglied"),
            "prozessglied": config.get("regelungsvariante", "Prozess"),
            "speicher": config.get("speicher", "Prozessspeicher"),
            "regelgroesse": f"Soll {setpoint:.4g} {unit}",
        }
        result["begruendung"].append(
            "Für diese Spezialanwendung wird ein transparentes, frei parametrierbares Ersatzmodell verwendet."
        )

    if "ks" in result:
        # Einfache, robuste IMC-nahe Startauslegung für ein PT1-Modell ohne Totzeit.
        result["kp"] = _clamp(2.0 / max(result["ks"], 0.001), 0.001, 100.0)
        result["ki"] = _clamp(result["kp"] / max(result["ts"], 0.1), 0.0, 100.0)
        result["dt"] = _clamp(result["t_end"] / 5000.0, 0.01, 60.0)
        result["u_min"] = float(config.get("stellgroesse_min", 0.0))
        result["u_max"] = float(config.get("stellgroesse_max", 100.0))

    return result


def derive_controller_from_wirkplan(config: dict):
    prozessart = config["prozessart"]
    traegheit = config["traegheit"]
    ueberschwingen_zulaessig = config["ueberschwingen_zulaessig"]
    bleibende_abweichung_erlaubt = config["bleibende_abweichung_erlaubt"]
    stoerungen_relevant = config["stoerungen_relevant"]

    result = {
        "controller_type": "PI",
        "plant_type": "PT1",
        "kp": 2.0,
        "ki": 0.4,
        "kd": 0.0,
        "ks": 1.0,
        "ts": 3.0,
        "zeta": 0.7,
        "omega0": 2.0,
        "setpoint": 1.0,
        "t_end": 25.0,
        "dt": 0.01,
        "disturbance_time": 10.0,
        "disturbance_value": -0.3,
        "disturbance_position": "Keine Störung",
        "begruendung": [],
    }

    if prozessart == "Temperaturregelung":
        result["plant_type"] = "PT1"
        result["controller_type"] = "PI"
        result["kp"] = 1.8
        result["ki"] = 0.25
        result["kd"] = 0.0
        result["ts"] = 8.0
        result["t_end"] = 50.0
        result["begruendung"].append(
            "Temperaturstrecken sind meistens träge PT1-ähnliche Strecken mit thermischer Speichermasse."
        )
        result["begruendung"].append(
            "Ein PI-Regler ist sinnvoll, weil ein reiner P-Regler häufig eine bleibende Temperaturabweichung hinterlässt."
        )

    elif prozessart == "Drehzahlregelung":
        result["plant_type"] = "PT1"
        result["controller_type"] = "PI"
        result["kp"] = 2.5
        result["ki"] = 0.7
        result["kd"] = 0.0
        result["ts"] = 2.0
        result["t_end"] = 20.0
        result["begruendung"].append(
            "Drehzahlstrecken besitzen mechanische Trägheit und lassen sich vereinfacht oft als PT1-Strecke betrachten."
        )
        result["begruendung"].append(
            "Ein PI-Regler reduziert die bleibende Drehzahlabweichung bei Laständerungen."
        )

    elif prozessart == "Füllstandsregelung":
        result["plant_type"] = "PT1"
        result["controller_type"] = "PI"
        result["kp"] = 1.5
        result["ki"] = 0.25
        result["kd"] = 0.0
        result["ts"] = 10.0
        result["t_end"] = 60.0
        result["begruendung"].append(
            "Füllstandsprozesse reagieren meist träge und besitzen ein speicherndes Verhalten."
        )
        result["begruendung"].append(
            "Ein PI-Regler ist geeignet, um den Füllstand trotz Zu- oder Abflussstörungen auf Sollwert zu bringen."
        )

    elif prozessart == "Druckregelung":
        result["plant_type"] = "PT1"
        result["controller_type"] = "PI"
        result["kp"] = 2.0
        result["ki"] = 0.5
        result["kd"] = 0.0
        result["ts"] = 3.0
        result["t_end"] = 25.0
        result["begruendung"].append(
            "Druckregelungen verhalten sich häufig wie mittelträge PT1-Strecken."
        )
        result["begruendung"].append(
            "Ein PI-Regler kann stationäre Druckabweichungen infolge von Verbrauch oder Leckage ausregeln."
        )

    elif prozessart == "Durchflussregelung":
        result["plant_type"] = "PT1"
        result["controller_type"] = "P"
        result["kp"] = 1.5
        result["ki"] = 0.0
        result["kd"] = 0.0
        result["ts"] = 1.0
        result["t_end"] = 12.0
        result["begruendung"].append(
            "Durchflussstrecken sind oft vergleichsweise schnell."
        )
        result["begruendung"].append(
            "Für einfache Betrachtungen kann ein P-Regler ausreichend sein; bei stationärer Abweichung ist PI sinnvoll."
        )

    elif prozessart == "Position / Mechanik":
        result["plant_type"] = "PT2"
        result["controller_type"] = "PID"
        result["kp"] = 3.0
        result["ki"] = 0.4
        result["kd"] = 0.4
        result["zeta"] = 0.55
        result["omega0"] = 2.2
        result["t_end"] = 20.0
        result["begruendung"].append(
            "Mechanische Positionssysteme können schwingfähig sein und werden deshalb vereinfacht als PT2-Strecke abgebildet."
        )
        result["begruendung"].append(
            "Ein PID-Regler kann Überschwingen dämpfen und gleichzeitig stationäre Abweichungen verringern."
        )

    physical = calculate_real_process_data(config)
    result["physical"] = physical

    uses_real_model = physical.get("active") and physical.get("supported") and "ks" in physical

    if uses_real_model:
        result["plant_type"] = physical.get("plant_type", "PT1")
        result["controller_type"] = "PI"
        result["kp"] = physical["kp"]
        result["ki"] = physical["ki"]
        result["kd"] = 0.0
        result["ks"] = physical["ks"]
        result["ts"] = physical["ts"]
        result["zeta"] = physical.get("zeta", result["zeta"])
        result["omega0"] = physical.get("omega0", result["omega0"])
        result["setpoint"] = physical["setpoint"]
        result["t_end"] = physical["t_end"]
        result["dt"] = physical["dt"]
        result["disturbance_value"] = physical["disturbance_value"]
        result["begruendung"].extend(physical["begruendung"])
        result["begruendung"].append(
            "Strecken- und Reglerparameter werden aus den realen Anlagendaten statt nur aus einer qualitativen Einschätzung gebildet."
        )

    if not uses_real_model and traegheit == "sehr träge":
        result["ts"] *= 1.8
        result["t_end"] *= 1.5
        result["kp"] *= 0.8
        result["ki"] *= 0.7
        result["begruendung"].append(
            "Da der Prozess als sehr träge bewertet wurde, werden die Startparameter vorsichtiger gewählt."
        )

    elif not uses_real_model and traegheit == "schnell":
        result["ts"] *= 0.5
        result["t_end"] *= 0.7
        result["kp"] *= 1.2
        result["ki"] *= 1.1
        result["begruendung"].append(
            "Da der Prozess als schnell bewertet wurde, wird eine kürzere Zeitkonstante angesetzt."
        )

    if bleibende_abweichung_erlaubt == "Nein":
        if result["controller_type"] == "P":
            result["controller_type"] = "PI"
            result["ki"] = 0.35
        result["begruendung"].append(
            "Da keine bleibende Regelabweichung erlaubt ist, wird mindestens ein PI-Regler empfohlen."
        )

    if ueberschwingen_zulaessig == "Nein":
        result["kp"] *= 0.8
        result["ki"] *= 0.8
        if result["controller_type"] == "PID":
            result["kd"] *= 1.2
        result["begruendung"].append(
            "Da Überschwingen nicht zulässig ist, werden die Reglerparameter defensiver gewählt."
        )

    if stoerungen_relevant == "Ja":
        result["disturbance_position"] = config.get("stoerort", "Vor der Strecke")
        result["disturbance_time"] = _clamp(
            config.get("stoerzeit_s", result["t_end"] / 2), 0.0, result["t_end"]
        )
        result["disturbance_value"] = float(config.get("stoerwert", result["disturbance_value"]))
        result["begruendung"].append(
            "Da relevante Störungen auftreten, wird eine Laststörung vor der Strecke für die Simulation vorgeschlagen."
        )
    else:
        result["disturbance_position"] = "Keine Störung"
        result["disturbance_time"] = 0.0
        result["disturbance_value"] = 0.0

    if config.get("auslegung") == "Manuell":
        result.update({
            "controller_type": config.get("man_controller_type", "PI"),
            "plant_type": config.get("man_plant_type", "PT1"),
            "kp": max(float(config.get("man_kp", result["kp"])), 0.0),
            "ki": max(float(config.get("man_ki", result["ki"])), 0.0),
            "kd": max(float(config.get("man_kd", result["kd"])), 0.0),
            "ks": max(float(config.get("man_ks", result["ks"])), 0.000001),
            "ts": max(float(config.get("man_ts", result["ts"])), 0.000001),
            "zeta": max(float(config.get("man_zeta", result["zeta"])), 0.01),
            "omega0": max(float(config.get("man_omega0", result["omega0"])), 0.000001),
            "setpoint": float(config.get("man_setpoint", result["setpoint"])),
            "t_end": max(float(config.get("man_t_end", result["t_end"])), 0.1),
            "dt": max(float(config.get("man_dt", result["dt"])), 0.0001),
        })
        result["begruendung"].append(
            "Die automatisch abgeleitete Auslegung wurde durch die manuellen Expertenwerte ersetzt."
        )

    selected_strategy = config.get("regelstrategie", "Automatische Empfehlung")
    result["regelstrategie"] = selected_strategy
    result["simulation_equivalent"] = result["controller_type"]
    if config.get("auslegung") != "Manuell" and selected_strategy != "Automatische Empfehlung":
        strategy_equivalents = {
            "P": "P", "PI": "PI", "PID": "PID", "Zweipunkt": "P",
            "Kaskade": "PI", "Split-Range": "PI", "Stufen-/Kaskadensteuerung": "PI",
            "Sicherheitsbegrenzung": "P", "Heizkurve + PI": "PI",
            "Begrenzungsregelung": "PI", "Dreipunktregelung": "PI",
            "Verhältnisregelung": "PI", "Prioritätssteuerung": "PI",
            "Energiemanagement": "PI", "Leistungsregelung": "PI",
        }
        equivalent = strategy_equivalents.get(selected_strategy, "PI")
        result["controller_type"] = equivalent
        result["simulation_equivalent"] = equivalent
        if equivalent == "P":
            result["ki"] = 0.0
            result["kd"] = 0.0
        elif equivalent == "PI":
            result["ki"] = max(float(result["ki"]), 0.001)
            result["kd"] = 0.0
        else:
            result["ki"] = max(float(result["ki"]), 0.001)
            result["kd"] = max(float(result["kd"]), 0.001)
        if selected_strategy not in {"P", "PI", "PID"}:
            result["begruendung"].append(
                f"Die Praxisstrategie „{selected_strategy}“ wird im vorhandenen Ein-Kreis-Simulator "
                f"durch einen {equivalent}-Regler angenähert; der Wirkplan behält die Fachbezeichnung."
            )

    result["kp"] = round(float(result["kp"]), 3)
    result["ki"] = round(float(result["ki"]), 8)
    result["kd"] = round(float(result["kd"]), 3)
    result["ks"] = round(float(result["ks"]), 6)
    result["ts"] = round(float(result["ts"]), 3)
    result["t_end"] = round(float(result["t_end"]), 3)
    result["dt"] = round(float(result["dt"]), 4)
    result["setpoint"] = round(float(result["setpoint"]), 4)
    result["disturbance_time"] = round(float(result["disturbance_time"]), 3)
    result["disturbance_value"] = round(float(result["disturbance_value"]), 4)

    return result


def build_wirkplan_flow(config: dict):
    physical = calculate_real_process_data(config)
    details = physical.get("node_details", {})

    def node_content(title: str, config_value: str, detail_key: str) -> str:
        detail = details.get(detail_key)
        if detail:
            return f"{title}<br>{config_value}<br><b>{detail}</b>"
        return f"{title}<br>{config_value}"

    derived = derive_controller_from_wirkplan(config)
    nodes = [
        StreamlitFlowNode(
            id="sollwert", pos=(0, 180),
            data={"content": f"Führungsgröße w<br>{config['sollwertgeber']}<br><b>{derived['setpoint']}</b>"},
            node_type="input", source_position="right", draggable=True
        ),
        StreamlitFlowNode(
            id="vergleich", pos=(240, 180), data={"content": "Vergleichsstelle<br>e = w − x"},
            node_type="default", source_position="right", target_position="left", draggable=True
        ),
        StreamlitFlowNode(
            id="regler", pos=(500, 180),
            data={"content": f"Regelstrategie<br><b>{config.get('regelstrategie', derived['controller_type'])}</b><br>Simulation: {derived['controller_type']}<br>Kp={derived['kp']}, Ki={derived['ki']}, Kd={derived['kd']}"},
            node_type="default", source_position="right", target_position="left", draggable=True
        ),
        StreamlitFlowNode(
            id="stellglied", pos=(800, 180),
            data={"content": f"Stellglied<br>{config['stellglied']}<br><b>{config['stellgroesse']}</b>"},
            node_type="default", source_position="right", target_position="left", draggable=True
        ),
        StreamlitFlowNode(
            id="stellgroesse",
            pos=(1090, 180),
            data={"content": node_content("Stellgröße", config["stellgroesse"], "stellgroesse")},
            node_type="input",
            source_position="right",
            draggable=True
        ),
        StreamlitFlowNode(
            id="prozessglied",
            pos=(1380, 180),
            data={"content": node_content("Prozessglied", config["prozessglied"], "prozessglied")},
            node_type="default",
            source_position="right",
            target_position="left",
            draggable=True
        ),
        StreamlitFlowNode(
            id="speicher",
            pos=(1680, 180),
            data={"content": node_content("Speicher / Trägheit", config["speicher"], "speicher")},
            node_type="default",
            source_position="right",
            target_position="left",
            draggable=True
        ),
        StreamlitFlowNode(
            id="regelgroesse",
            pos=(1980, 180),
            data={"content": node_content("Regelgröße", config["regelgroesse"], "regelgroesse")},
            node_type="output",
            target_position="left",
            draggable=True
        ),
        StreamlitFlowNode(
            id="sensor", pos=(1680, 400),
            data={"content": f"Sensor / Messglied<br>{config['sensor']}<br>{config['messglied']}"},
            node_type="default", source_position="left", target_position="right", draggable=True
        ),
    ]

    edges = [
        StreamlitFlowEdge(id="r1", source="sollwert", target="vergleich", animated=True, label="w"),
        StreamlitFlowEdge(id="r2", source="vergleich", target="regler", animated=True, label="e"),
        StreamlitFlowEdge(id="r3", source="regler", target="stellglied", animated=True, label="u"),
        StreamlitFlowEdge(id="r4", source="stellglied", target="stellgroesse", animated=True, label="Stellsignal"),
        StreamlitFlowEdge(
            id="w1",
            source="stellgroesse",
            target="prozessglied",
            animated=True,
            label="wirkt auf"
        ),
        StreamlitFlowEdge(id="r5", source="regelgroesse", target="sensor", animated=True, label="x"),
        StreamlitFlowEdge(id="r6", source="sensor", target="vergleich", animated=True, label="Rückführung (−)"),
        StreamlitFlowEdge(
            id="w2",
            source="prozessglied",
            target="speicher",
            animated=True,
            label="Energie / Stoff"
        ),
        StreamlitFlowEdge(
            id="w3",
            source="speicher",
            target="regelgroesse",
            animated=True,
            label="Messgröße"
        ),
    ]

    if config["stoerungen_relevant"] == "Ja":
        nodes.append(
            StreamlitFlowNode(
                id="stoerung",
                pos=(1380, 20),
                data={"content": f"Störgröße<br>{config['stoergroesse']}"},
                node_type="input",
                source_position="right",
                draggable=True
            )
        )

        edges.append(
            StreamlitFlowEdge(
                id="w4",
                source="stoerung",
                target="prozessglied" if config.get("stoerort") == "Vor der Strecke" else "regelgroesse",
                animated=True,
                label="Störeinfluss"
            )
        )

    return StreamlitFlowState(nodes, edges)


def update_wirkplan_defaults_for_process(config: dict):
    """Aktualisiert Prozess-Presets und synchronisiert zugleich die sichtbaren Widgets."""
    prozessart = config["prozessart"]

    presets = {
        "Temperaturregelung": {
            "stellgroesse": "Heizleistung P [W]",
            "prozessglied": "Heizkörper / Wärmeerzeuger",
            "speicher": "thermische Masse des Raums",
            "regelgroesse": "Raumtemperatur T [°C]",
            "stoergroesse": "Außentemperatur / geöffnete Tür",
            "traegheit": "sehr träge",
            "ueberschwingen_zulaessig": "Nein",
            "bleibende_abweichung_erlaubt": "Nein",
            "stoerungen_relevant": "Ja",
        },
        "Drehzahlregelung": {
            "stellgroesse": "Motorspannung U [V]",
            "prozessglied": "Motor / Umrichter",
            "speicher": "mechanische Trägheit J",
            "regelgroesse": "Drehzahl n [1/min]",
            "stoergroesse": "Lastmoment M_L",
            "traegheit": "mittel",
            "ueberschwingen_zulaessig": "Ja",
            "bleibende_abweichung_erlaubt": "Nein",
            "stoerungen_relevant": "Ja",
        },
        "Füllstandsregelung": {
            "stellgroesse": "Ventilöffnung [%]",
            "prozessglied": "Zulaufventil / Pumpe",
            "speicher": "Behältervolumen",
            "regelgroesse": "Füllstand h [m]",
            "stoergroesse": "Abfluss / Verbrauch",
            "traegheit": "sehr träge",
            "ueberschwingen_zulaessig": "Nein",
            "bleibende_abweichung_erlaubt": "Nein",
            "stoerungen_relevant": "Ja",
        },
        "Druckregelung": {
            "stellgroesse": "Pumpendrehzahl / Ventilstellung",
            "prozessglied": "Pumpe / Rohrnetz",
            "speicher": "Kompressibilität / Leitungsvolumen",
            "regelgroesse": "Druck p [bar]",
            "stoergroesse": "Verbrauch / Leckage",
            "traegheit": "mittel",
            "ueberschwingen_zulaessig": "Nein",
            "bleibende_abweichung_erlaubt": "Nein",
            "stoerungen_relevant": "Ja",
        },
        "Durchflussregelung": {
            "stellgroesse": "Ventilöffnung [%]",
            "prozessglied": "Ventil / Rohrstrecke",
            "speicher": "geringe Prozessspeicherung",
            "regelgroesse": "Durchfluss q [m³/h]",
            "stoergroesse": "Vordruckschwankung",
            "traegheit": "schnell",
            "ueberschwingen_zulaessig": "Ja",
            "bleibende_abweichung_erlaubt": "Ja",
            "stoerungen_relevant": "Nein",
        },
        "Position / Mechanik": {
            "stellgroesse": "Motorspannung / Stellmoment",
            "prozessglied": "Antrieb / Mechanik",
            "speicher": "Masse, Feder, Trägheit",
            "regelgroesse": "Position x [mm]",
            "stoergroesse": "Lastkraft / Reibung",
            "traegheit": "mittel",
            "ueberschwingen_zulaessig": "Nein",
            "bleibende_abweichung_erlaubt": "Nein",
            "stoerungen_relevant": "Ja",
        },
    }

    if prozessart in presets:
        widget_keys = {
            "stellgroesse": "wirkplan_stellgroesse",
            "prozessglied": "wirkplan_prozessglied",
            "speicher": "wirkplan_speicher",
            "regelgroesse": "wirkplan_regelgroesse",
            "stoergroesse": "wirkplan_stoergroesse",
            "traegheit": "wirkplan_traegheit",
            "ueberschwingen_zulaessig": "wirkplan_ueberschwingen",
            "bleibende_abweichung_erlaubt": "wirkplan_abweichung",
            "stoerungen_relevant": "wirkplan_stoerungen",
        }

        for key, value in presets[prozessart].items():
            config[key] = value
            widget_key = widget_keys.get(key)
            if widget_key is not None:
                st.session_state[widget_key] = value

        chain_presets = {
            "Temperaturregelung": ("Leistungssteller / Heizung", "Temperaturfühler", "Messumformer"),
            "Drehzahlregelung": ("Frequenzumrichter", "Drehgeber", "Drehzahlmessumformer"),
            "Füllstandsregelung": ("Stellventil / Pumpe", "Füllstandssensor", "Messumformer"),
            "Druckregelung": ("Verdichter / Stellventil", "Drucksensor", "Druckmessumformer"),
            "Durchflussregelung": ("Regelventil", "Durchflusssensor", "Durchflussmessumformer"),
            "Position / Mechanik": ("Servoantrieb", "Positionsgeber", "Positionsmessumformer"),
        }
        stellglied, sensor, messglied = chain_presets[prozessart]
        for key, value in {
            "stellglied": stellglied, "sensor": sensor, "messglied": messglied
        }.items():
            config[key] = value
            st.session_state[f"wirkplan_{key}"] = value

    return config


def apply_practical_variant_defaults(config: dict):
    """Übernimmt Fachpreset, Stellgliedauswahl und Grundmodell in den Wirkplan."""
    category = config.get("anlagenart", "RLT / Lüftung")
    variants = PRACTICAL_PROCESS_CATALOG[category]
    variant_name = config.get("regelungsvariante")
    if variant_name not in variants:
        variant_name = next(iter(variants))
        config["regelungsvariante"] = variant_name

    profile = variants[variant_name]
    config["prozessart"] = profile["model"]
    config = update_wirkplan_defaults_for_process(config)

    actuator = config.get("stellglied_typ")
    if actuator not in profile["actuators"]:
        actuator = profile["actuators"][0]
    strategy = config.get("regelstrategie")
    if strategy not in profile["strategies"]:
        strategy = profile["strategies"][0]

    config.update({
        "stellglied_typ": actuator,
        "regelstrategie": strategy,
        "stellglied": actuator,
        "stellgroesse": f"Stellsignal an {actuator}",
        "prozessglied": profile["process"],
        "speicher": profile["storage"],
        "regelgroesse": profile["controlled"],
        "stoergroesse": profile["disturbance"],
        "sensor": f"Sensor für {profile['controlled'].split(' [')[0]}",
        "messglied": "Messumformer / Automationsstation",
        "traegheit": profile["inertia"],
        "ueberschwingen_zulaessig": profile["overshoot"],
        "bleibende_abweichung_erlaubt": profile["offset"],
        "stoerungen_relevant": profile["disturbances"],
        "auslegung": "Automatisch",
        "reale_daten_aktiv": True,
    })

    controlled = profile["controlled"]
    model = profile["model"]

    # Praxiswerte statt eines einzigen Universal-Presets. Die Werte sind
    # bewusst plausible Startpunkte und bleiben in der Oberfläche editierbar.
    if model == "Temperaturregelung":
        if category in {"RLT / Lüftung", "Raumautomation"}:
            config.update({
                "temp_betriebsart": "Kühlen" if "Kühl" in variant_name else "Heizen",
                "temp_medium": "Luft", "temp_volumen_m3": 300.0,
                "temp_heizleistung_kw": 20.0, "temp_umgebung_c": 5.0,
                "temp_soll_c": 21.0, "temp_waermeverlust_w_k": 600.0,
                "temp_wirkungsgrad": 0.9,
            })
        elif category == "Kälte":
            config.update({
                "temp_betriebsart": "Kühlen", "temp_medium": "Wasser",
                "temp_volumen_m3": 0.8, "temp_heizleistung_kw": 35.0,
                "temp_umgebung_c": 12.0, "temp_soll_c": 6.0,
                "temp_waermeverlust_w_k": 900.0, "temp_wirkungsgrad": 0.85,
            })
        elif category in {"Dampf", "Prozesswärme"}:
            config.update({
                "temp_betriebsart": "Heizen", "temp_medium": "Benutzerdefiniert",
                "temp_volumen_m3": 2.0, "temp_heizleistung_kw": 150.0,
                "temp_umgebung_c": 20.0, "temp_soll_c": 180.0,
                "temp_waermeverlust_w_k": 750.0, "temp_wirkungsgrad": 0.88,
                "temp_dichte_kg_m3": 780.0, "temp_cp_kj_kgk": 0.6,
            })
        else:
            config.update({
                "temp_betriebsart": "Heizen", "temp_medium": "Wasser",
                "temp_volumen_m3": 0.5, "temp_heizleistung_kw": 30.0,
                "temp_umgebung_c": 20.0, "temp_soll_c": 55.0,
                "temp_waermeverlust_w_k": 450.0, "temp_wirkungsgrad": 0.95,
            })
    elif model == "Druckregelung":
        if category == "Druckluft":
            config.update({
                "druck_volumen_m3": 2.0, "druck_max_bar": 10.0,
                "druck_soll_bar": 7.0, "druck_foerderstrom_m3h": 180.0,
                "druck_verbrauch_m3h": 90.0, "druck_zeitkonstante_s": 4.0,
            })
        elif category == "Dampf":
            config.update({
                "druck_volumen_m3": 5.0, "druck_max_bar": 16.0,
                "druck_soll_bar": 10.0, "druck_foerderstrom_m3h": 300.0,
                "druck_verbrauch_m3h": 180.0, "druck_zeitkonstante_s": 8.0,
            })
        else:
            config.update({
                "druck_volumen_m3": 1.0, "druck_max_bar": 10.0,
                "druck_soll_bar": 5.0, "druck_foerderstrom_m3h": 80.0,
                "druck_verbrauch_m3h": 35.0, "druck_zeitkonstante_s": 3.0,
            })
    elif model == "Durchflussregelung":
        if category in {"RLT / Lüftung", "Raumautomation"}:
            config.update({
                "flow_max_m3h": 10000.0, "flow_soll_m3h": 6000.0,
                "flow_rohrlaenge_m": 30.0, "flow_durchmesser_mm": 500.0,
                "flow_ventilzeit_s": 1.5, "flow_druckverlust_bar": 0.004,
            })
        elif category in {"Dosierung / Chemie", "Abwasser"} and "Dosier" in variant_name:
            config.update({
                "flow_max_m3h": 0.1, "flow_soll_m3h": 0.05,
                "flow_rohrlaenge_m": 5.0, "flow_durchmesser_mm": 10.0,
                "flow_ventilzeit_s": 0.5, "flow_druckverlust_bar": 2.0,
            })
        else:
            config.update({
                "flow_max_m3h": 100.0, "flow_soll_m3h": 60.0,
                "flow_rohrlaenge_m": 25.0, "flow_durchmesser_mm": 80.0,
                "flow_ventilzeit_s": 1.0, "flow_druckverlust_bar": 1.5,
            })
    elif model == "Füllstandsregelung":
        config.update({
            "tank_volumen_m3": 5.0 if category != "Dampf" else 2.0,
            "tank_hoehe_m": 2.5, "tank_zulauf_m3h": 12.0,
            "tank_abfluss_m3h": 6.0, "tank_soll_m": 1.5,
        })
    elif model == "Drehzahlregelung":
        config.update({
            "motor_leistung_kw": 7.5, "motor_nenndrehzahl_rpm": 1500.0,
            "motor_soll_rpm": 1200.0, "motor_hochlaufzeit_s": 4.0,
            "motor_spannung_v": 400.0, "motor_wirkungsgrad": 0.9,
            "motor_traegheit_kgm2": 0.25, "motor_lastmoment_nm": 30.0,
        })
    elif model == "Position / Mechanik":
        config.update({
            "pos_masse_kg": 40.0, "pos_feder_n_m": 1500.0,
            "pos_daempfung_ns_m": 220.0, "pos_stellkraft_n": 2000.0,
            "pos_hub_mm": 500.0, "pos_soll_mm": 250.0,
        })

    if profile["model"] == "Generische Prozessstrecke":
        generic_defaults = {
            "CO₂": (1000.0, "ppm", 300.0),
            "pH": (7.0, "pH", 120.0),
            "Sauerstoff": (2.0, "mg/l", 300.0),
            "Feuchte": (50.0, "% r. F.", 300.0),
            "Leitfähigkeit": (500.0, "µS/cm", 300.0),
            "Konzentration": (50.0, "%", 300.0),
            "Ladezustand": (80.0, "%", 3600.0),
            "Leistung": (100.0, "kW", 300.0),
            "Netzleistung": (0.0, "kW", 300.0),
            "Taupunkt": (-20.0, "°C", 600.0),
            "Differenzdruck": (250.0, "Pa", 60.0),
            "Drehmoment": (50.0, "Nm", 30.0),
            "Kraft": (1000.0, "N", 30.0),
            "Mischungsverhältnis": (1.0, "Verhältnis", 60.0),
            "Gesamtleistung": (250.0, "kW", 300.0),
            "Überhitzung": (6.0, "K", 60.0),
        }
        setpoint, unit, t_end = 1.0, "Prozesseinheit", 60.0
        for token, values in generic_defaults.items():
            if token in controlled:
                setpoint, unit, t_end = values
                break
        config.update({
            "generic_setpoint": setpoint,
            "generic_unit": unit,
            "generic_t_end": t_end,
            "generic_ts": max(t_end / 6.0, 0.1),
        })

    widget_values = {
        "wirkplan_stellglied": config["stellglied"],
        "wirkplan_stellgroesse": config["stellgroesse"],
        "wirkplan_prozessglied": config["prozessglied"],
        "wirkplan_speicher": config["speicher"],
        "wirkplan_regelgroesse": config["regelgroesse"],
        "wirkplan_stoergroesse": config["stoergroesse"],
        "wirkplan_sensor": config["sensor"],
        "wirkplan_messglied": config["messglied"],
        "wirkplan_traegheit": config["traegheit"],
        "wirkplan_ueberschwingen": config["ueberschwingen_zulaessig"],
        "wirkplan_abweichung": config["bleibende_abweichung_erlaubt"],
        "wirkplan_stoerungen": config["stoerungen_relevant"],
    }
    # Alte Widgetzustände entfernen, damit die neuen Fachpresets beim Rerun
    # als eindeutige Defaults erscheinen und nicht von alten Eingaben überlagert werden.
    for key in widget_values:
        st.session_state.pop(key, None)
    return config


def reset_physical_wirkplan():
    """Setzt Auswahl, Prozessdaten und Widgetzustände des Wirkplan-Builders zurück."""
    fresh = ensure_real_process_defaults({
        "anlagenart": "RLT / Lüftung",
        "regelungsvariante": "Zulufttemperatur",
        "stellglied_typ": "2-Wege-Regelventil",
        "regelstrategie": "Automatische Empfehlung",
        "prozessart": "Temperaturregelung",
        "stellgroesse": "Stellsignal an 2-Wege-Regelventil",
        "prozessglied": "Heiz-/Kühlregister",
        "speicher": "Luft- und Registermasse",
        "regelgroesse": "Zulufttemperatur [°C]",
        "stoergroesse": "Außenluft / Last",
        "traegheit": "mittel",
        "ueberschwingen_zulaessig": "Nein",
        "bleibende_abweichung_erlaubt": "Nein",
        "stoerungen_relevant": "Ja",
    })
    fresh = apply_practical_variant_defaults(fresh)
    st.session_state.wirkplan_config = fresh
    st.session_state.clear_wirkplan_widgets_pending = True
    st.session_state.parameter_sync_notice = "Physikalischer Wirkplan wurde auf die Praxis-Standardwerte zurückgesetzt."


def render_real_process_inputs(config: dict):
    """Zeigt nur die zur Prozessart und Eingabetiefe passenden realen Anlagendaten."""
    config["reale_daten_aktiv"] = st.checkbox(
        "Reale Anlagendaten für die Berechnung verwenden",
        value=bool(config.get("reale_daten_aktiv", True)),
        key="wirkplan_reale_daten_aktiv",
    )

    if not config["reale_daten_aktiv"]:
        st.caption("Die App verwendet wieder die qualitativen Prozess-Presets.")
        return config

    config["eingabetiefe"] = st.selectbox(
        "Eingabetiefe",
        ["Einfach", "Erweitert", "Experte"],
        index=["Einfach", "Erweitert", "Experte"].index(
            config.get("eingabetiefe", "Einfach")
        ),
        key="wirkplan_eingabetiefe",
        help="Ausgeblendete Detailwerte bleiben als gekennzeichnete Standardannahmen aktiv.",
    )
    tiefe = config["eingabetiefe"]
    prozessart = config["prozessart"]

    def number(key, label, minimum, step, help_text=None, fmt=None):
        kwargs = {
            "label": label,
            "min_value": float(minimum),
            "value": float(config[key]),
            "step": float(step),
            "key": f"wirkplan_{key}",
            "help": help_text,
        }
        if fmt is not None:
            kwargs["format"] = fmt
        config[key] = st.number_input(**kwargs)

    if prozessart == "Temperaturregelung":
        config["temp_betriebsart"] = st.selectbox(
            "Thermische Betriebsart",
            ["Heizen", "Kühlen"],
            index=["Heizen", "Kühlen"].index(config.get("temp_betriebsart", "Heizen")),
            key="wirkplan_temp_betriebsart",
            help="Legt fest, ob die Anlage Wärme zuführt oder dem Prozess Wärme entzieht.",
        )
        config["temp_medium"] = st.selectbox(
            "Medium",
            ["Wasser", "Luft", "Benutzerdefiniert"],
            index=["Wasser", "Luft", "Benutzerdefiniert"].index(
                config.get("temp_medium", "Wasser")
            ),
            key="wirkplan_temp_medium",
        )
        number("temp_volumen_m3", "Volumen [m³]", 0.001, 0.1, fmt="%.3f")
        number("temp_heizleistung_kw", "Heizleistung [kW]", 0.001, 0.5, fmt="%.3f")
        number("temp_umgebung_c", "Umgebungstemperatur [°C]", -100.0, 1.0)
        number("temp_soll_c", "Solltemperatur [°C]", -100.0, 1.0)

        if tiefe in ["Erweitert", "Experte"]:
            number(
                "temp_waermeverlust_w_k",
                "Wärmeverlustkoeffizient [W/K]",
                0.1,
                10.0,
                "Wärmeleistung, die je Kelvin Temperaturdifferenz an die Umgebung verloren geht.",
            )
        else:
            st.caption(
                f"Annahme: Wärmeverlustkoeffizient {config['temp_waermeverlust_w_k']:.0f} W/K."
            )

        if tiefe == "Experte":
            number("temp_wirkungsgrad", "Wirkungsgrad [0–1]", 0.01, 0.01, fmt="%.2f")
            if config["temp_medium"] == "Benutzerdefiniert":
                number("temp_dichte_kg_m3", "Dichte [kg/m³]", 0.001, 1.0, fmt="%.3f")
                number(
                    "temp_cp_kj_kgk",
                    "spezifische Wärmekapazität [kJ/(kg·K)]",
                    0.001,
                    0.01,
                    fmt="%.3f",
                )
        else:
            st.caption(f"Annahme: thermischer Wirkungsgrad {config['temp_wirkungsgrad'] * 100:.0f} %.")

    elif prozessart == "Drehzahlregelung":
        number("motor_leistung_kw", "Motorleistung [kW]", 0.01, 0.1)
        number("motor_nenndrehzahl_rpm", "Nenndrehzahl [1/min]", 1.0, 10.0)
        number("motor_soll_rpm", "Solldrehzahl [1/min]", 0.0, 10.0)
        number("motor_hochlaufzeit_s", "vorgegebene Hochlaufzeit [s]", 0.1, 0.5)

        if tiefe in ["Erweitert", "Experte"]:
            number("motor_leitung_m", "Motorleitung – einfache Länge [m]", 0.0, 5.0)
            number("motor_querschnitt_mm2", "Leiterquerschnitt [mm²]", 0.1, 0.5)
        else:
            st.caption(
                f"Annahme: {config['motor_leitung_m']:.0f} m Motorleitung mit "
                f"{config['motor_querschnitt_mm2']:.1f} mm² Cu."
            )

        if tiefe == "Experte":
            number("motor_spannung_v", "Drehspannung [V]", 1.0, 10.0)
            number("motor_wirkungsgrad", "Wirkungsgrad η [0–1]", 0.1, 0.01, fmt="%.2f")
            number("motor_leistungsfaktor", "Leistungsfaktor cos φ [0–1]", 0.1, 0.01, fmt="%.2f")
            number("motor_traegheit_kgm2", "Gesamtträgheitsmoment J [kg·m²]", 0.0001, 0.01, fmt="%.4f")
            number("motor_lastmoment_nm", "Lastmoment [Nm]", 0.0, 1.0)
        else:
            st.caption(
                f"Annahmen: η {config['motor_wirkungsgrad']:.2f}, cos φ "
                f"{config['motor_leistungsfaktor']:.2f}, J {config['motor_traegheit_kgm2']:.3f} kg·m², "
                f"Lastmoment {config['motor_lastmoment_nm']:.1f} Nm."
            )

    elif prozessart == "Füllstandsregelung":
        number("tank_volumen_m3", "Behältervolumen [m³]", 0.001, 0.1, fmt="%.3f")
        number("tank_hoehe_m", "maximale Füllhöhe [m]", 0.01, 0.1)
        number("tank_zulauf_m3h", "maximaler Zulauf [m³/h]", 0.001, 0.5, fmt="%.3f")
        number("tank_soll_m", "Sollfüllstand [m]", 0.0, 0.1)

        if tiefe in ["Erweitert", "Experte"]:
            config["tank_form"] = st.selectbox(
                "Behälterform",
                ["Zylindrisch", "Rechteckig"],
                index=["Zylindrisch", "Rechteckig"].index(
                    config.get("tank_form", "Zylindrisch")
                ),
                key="wirkplan_tank_form",
            )
            number("tank_abfluss_m3h", "konstanter Abfluss [m³/h]", 0.0, 0.5)
        else:
            st.caption(
                f"Annahme: {config['tank_form'].lower()}er Behälter und "
                f"{config['tank_abfluss_m3h']:.1f} m³/h konstanter Abfluss."
            )

        if tiefe == "Experte":
            st.caption(
                "Das Modell verwendet aktuell einen konstanten Querschnitt und konstanten Abfluss. "
                "Ein höhenabhängiger freier Auslauf folgt in einer späteren Ausbaustufe."
            )

    elif prozessart == "Druckregelung":
        number("druck_volumen_m3", "Speichervolumen [m³]", 0.001, 0.1, fmt="%.3f")
        number("druck_max_bar", "Maximaldruck [bar]", 0.01, 0.5)
        number("druck_soll_bar", "Solldruck [bar]", 0.0, 0.1)
        number("druck_foerderstrom_m3h", "maximaler Förderstrom [m³/h]", 0.001, 1.0)
        if tiefe in ["Erweitert", "Experte"]:
            number("druck_verbrauch_m3h", "Grundverbrauch [m³/h]", 0.0, 1.0)
        if tiefe == "Experte":
            number("druck_zeitkonstante_s", "minimale Anlagenzeitkonstante [s]", 0.05, 0.1)

    elif prozessart == "Durchflussregelung":
        number("flow_max_m3h", "maximaler Durchfluss [m³/h]", 0.001, 1.0)
        number("flow_soll_m3h", "Solldurchfluss [m³/h]", 0.0, 1.0)
        number("flow_ventilzeit_s", "Ventil-Stellzeit [s]", 0.01, 0.1)
        if tiefe in ["Erweitert", "Experte"]:
            number("flow_rohrlaenge_m", "Rohrlänge [m]", 0.0, 1.0)
            number("flow_durchmesser_mm", "Rohr-Innendurchmesser [mm]", 1.0, 5.0)
        if tiefe == "Experte":
            number("flow_druckverlust_bar", "Druckverlust bei Nennfluss [bar]", 0.0, 0.1)

    elif prozessart == "Position / Mechanik":
        number("pos_masse_kg", "bewegte Masse [kg]", 0.001, 1.0)
        number("pos_stellkraft_n", "maximale Stellkraft [N]", 0.001, 50.0)
        number("pos_hub_mm", "maximaler Hub [mm]", 0.001, 10.0)
        number("pos_soll_mm", "Sollposition [mm]", 0.0, 10.0)
        if tiefe in ["Erweitert", "Experte"]:
            number("pos_feder_n_m", "Federsteifigkeit [N/m]", 0.001, 100.0)
            number("pos_daempfung_ns_m", "Dämpfung [N·s/m]", 0.0, 10.0)

    elif prozessart == "Generische Prozessstrecke":
        config["generic_plant_type"] = st.selectbox(
            "Ersatzmodell", ["PT1", "PT2"],
            index=["PT1", "PT2"].index(config.get("generic_plant_type", "PT1")),
            key="wirkplan_generic_plant_type",
        )
        number("generic_ks", "Streckenverstärkung Ks", 0.000001, 0.1, fmt="%.6f")
        number("generic_ts", "Zeitkonstante Ts [s]", 0.000001, 0.5, fmt="%.6f")
        number("generic_setpoint", "Sollwert", -1000000.0, 0.1)
        config["generic_unit"] = st.text_input(
            "Einheit der Regelgröße", value=config.get("generic_unit", "Prozesseinheit"),
            key="wirkplan_generic_unit",
        )
        number("generic_t_end", "Simulationsdauer [s]", 0.1, 1.0)
        if config["generic_plant_type"] == "PT2":
            number("generic_zeta", "Dämpfungsgrad ζ", 0.01, 0.05)
            number("generic_omega0", "Eigenkreisfrequenz ω0 [rad/s]", 0.000001, 0.1)

    return config


def render_wirkplan_builder():
    clear_pending_parameter_widgets()
    if st.session_state.pop("clear_wirkplan_widgets_pending", False):
        for key in list(st.session_state.keys()):
            if key.startswith("wirkplan_") and key != "wirkplan_config":
                del st.session_state[key]
    st.title("Physikalischer Wirkplan-Builder")

    st.caption(
        "Hier startest du nicht mit Regler und Strecke, sondern mit physikalischen Größen. "
        "Aus dem Wirkplan leitet die App ein geeignetes Streckenmodell und einen Startregler ab."
    )

    config = ensure_real_process_defaults(st.session_state.wirkplan_config)
    pending_start = st.session_state.pop("pending_start_wirkplan", None)
    if pending_start:
        # Eventuell vorhandene Widgetwerte dürfen die Auswahl aus dem
        # Startformular nicht wieder überschreiben.
        for key in [
            "wirkplan_anlagenart", "wirkplan_regelungsvariante",
            "wirkplan_stellglied_typ", "wirkplan_regelstrategie",
        ]:
            st.session_state.pop(key, None)
        config.update(pending_start)
        config = apply_practical_variant_defaults(config)
        st.session_state.wirkplan_config = config
        st.session_state.wirkplan_last_variant = config["regelungsvariante"]
        st.session_state.wirkplan_last_actuator = config["stellglied_typ"]

    col_left = st.sidebar
    col_right = st.container()

    with col_left:
        st.header("Bedienpanel")
        st.caption("Praxis- und Prozessoptionen links · Wirkplan rechts")

        if "parameter_sync_notice" in st.session_state:
            st.success(st.session_state.pop("parameter_sync_notice"))

        if st.button(
            "Parameter aus Simulation übernehmen",
            width="stretch",
            key="wirkplan_import_simulation",
            help="Übernimmt alle gemeinsamen Regler-, Strecken-, Sollwert- und Störparameter in die manuelle Auslegung.",
        ):
            apply_shared_parameters(shared_parameters_from_session(), "der Simulation")
            st.rerun()

        if st.button(
            "Wirkplan komplett zurücksetzen",
            width="stretch",
            key="wirkplan_reset_all",
            help="Setzt Praxisauswahl, Anlagendaten, Messkette, Störungen und Auslegung auf die Startwerte zurück.",
        ):
            reset_physical_wirkplan()
            st.rerun()

        st.subheader("Physikalische Angaben")

        st.markdown("#### Praxisauswahl")
        old_category = config.get("anlagenart", "RLT / Lüftung")
        categories = list(PRACTICAL_PROCESS_CATALOG)
        if old_category not in categories:
            old_category = categories[0]
        config["anlagenart"] = st.selectbox(
            "1. Anlagenart / Gewerk", categories, index=categories.index(old_category),
            key="wirkplan_anlagenart",
            help="Filtert den Praxiskatalog nach dem technischen Gewerk, zum Beispiel RLT, Heizung, Kälte oder Wasser.",
        )

        variants = PRACTICAL_PROCESS_CATALOG[config["anlagenart"]]
        variant_names = list(variants)
        old_variant = config.get("regelungsvariante")
        category_changed = config["anlagenart"] != old_category
        if category_changed or old_variant not in variant_names:
            old_variant = variant_names[0]
            st.session_state.pop("wirkplan_regelungsvariante", None)
        config["regelungsvariante"] = st.selectbox(
            "2. Regelungsvariante", variant_names, index=variant_names.index(old_variant),
            key="wirkplan_regelungsvariante",
            help="Bestimmt die konkrete Regelaufgabe und lädt passende Startwerte für Prozess, Sensorik und Störungen.",
        )

        profile = variants[config["regelungsvariante"]]
        actuator_options = profile["actuators"]
        old_actuator = config.get("stellglied_typ")
        if old_actuator not in actuator_options:
            old_actuator = actuator_options[0]
        config["stellglied_typ"] = st.selectbox(
            "3. Stellglied / Antrieb", actuator_options,
            index=actuator_options.index(old_actuator), key="wirkplan_stellglied_typ",
            help="Wählt das physikalische Stellglied, das das Reglerausgangssignal in den Prozess einbringt.",
        )

        strategy_options = profile["strategies"]
        old_strategy = config.get("regelstrategie")
        if old_strategy not in strategy_options:
            old_strategy = strategy_options[0]
        config["regelstrategie"] = st.selectbox(
            "4. Regelstrategie", strategy_options,
            index=strategy_options.index(old_strategy), key="wirkplan_regelstrategie",
            help="Legt das fachliche Regelkonzept fest. Erweiterte Strategien werden transparent auf den Ein-Kreis-Simulator abgebildet.",
        )

        selection_changed = (
            category_changed
            or config["regelungsvariante"] != st.session_state.get("wirkplan_last_variant")
        )
        if selection_changed:
            config = apply_practical_variant_defaults(config)
            st.session_state["wirkplan_last_variant"] = config["regelungsvariante"]
            st.session_state.wirkplan_config = config
            st.rerun()

        config["prozessart"] = profile["model"]
        if config["stellglied_typ"] != st.session_state.get("wirkplan_last_actuator"):
            config["stellglied"] = config["stellglied_typ"]
            config["stellgroesse"] = f"Stellsignal an {config['stellglied_typ']}"
            st.session_state["wirkplan_stellglied"] = config["stellglied"]
            st.session_state["wirkplan_stellgroesse"] = config["stellgroesse"]
            st.session_state["wirkplan_last_actuator"] = config["stellglied_typ"]
        st.caption(
            f"Grundmodell: **{config['prozessart']}** · "
            f"{len(variant_names)} Varianten in diesem Gewerk"
        )

        def config_text_input(label, field, widget_key):
            if widget_key not in st.session_state:
                st.session_state[widget_key] = str(config.get(field, ""))
            help_texts = {
                "stellgroesse": "Physikalische Größe, mit der das Stellglied auf den Prozess wirkt.",
                "prozessglied": "Anlagenteil, in dem Energie, Stoff oder Bewegung umgesetzt wird.",
                "speicher": "Bestimmt wesentlich Trägheit, Zeitkonstante und dynamisches Verhalten.",
                "regelgroesse": "Messbare Prozessgröße, die dem Sollwert folgen soll.",
                "stellglied": "Technisches Bauteil zwischen Regler und Prozess.",
                "sensor": "Erfasst die Regelgröße für die Rückführung.",
                "messglied": "Bereitet das Sensorsignal für Regler oder Automationsstation auf.",
                "sollwertgeber": "Quelle der gewünschten Führungsgröße.",
            }
            config[field] = st.text_input(
                label, key=widget_key, help=help_texts.get(field)
            )

        with st.expander("1. Physikalische Wirkungskette", expanded=True):
            config_text_input("Stellgröße", "stellgroesse", "wirkplan_stellgroesse")
            config_text_input("Prozessglied", "prozessglied", "wirkplan_prozessglied")
            config_text_input("Speicher / Trägheit", "speicher", "wirkplan_speicher")
            config_text_input("Regelgröße", "regelgroesse", "wirkplan_regelgroesse")
            config_text_input("Stellglied", "stellglied", "wirkplan_stellglied")
            config_text_input("Sensor", "sensor", "wirkplan_sensor")
            config_text_input("Messumformer / Messglied", "messglied", "wirkplan_messglied")
            config_text_input("Sollwertgeber", "sollwertgeber", "wirkplan_sollwertgeber")

        with st.expander("2. Reale Anlagendaten", expanded=True):
            config = render_real_process_inputs(config)

        with st.expander("3. Verhalten des Prozesses", expanded=False):
            config["traegheit"] = st.selectbox(
                "Wie träge ist der Prozess?",
                ["schnell", "mittel", "träge", "sehr träge"],
                index=["schnell", "mittel", "träge", "sehr träge"].index(config["traegheit"]),
                key="wirkplan_traegheit"
            )

            config["ueberschwingen_zulaessig"] = st.selectbox(
                "Ist Überschwingen zulässig?",
                ["Ja", "Nein"],
                index=["Ja", "Nein"].index(config["ueberschwingen_zulaessig"]),
                key="wirkplan_ueberschwingen"
            )

            config["bleibende_abweichung_erlaubt"] = st.selectbox(
                "Ist eine bleibende Regelabweichung erlaubt?",
                ["Ja", "Nein"],
                index=["Ja", "Nein"].index(config["bleibende_abweichung_erlaubt"]),
                key="wirkplan_abweichung"
            )

            config["totzeit_s"] = st.number_input(
                "Totzeit [s]", min_value=0.0, value=float(config["totzeit_s"]),
                step=0.1, key="wirkplan_totzeit_s"
            )
            limit_left, limit_right = st.columns(2)
            with limit_left:
                config["stellgroesse_min"] = st.number_input(
                    "Stellgröße min.", value=float(config["stellgroesse_min"]),
                    step=1.0, key="wirkplan_stellgroesse_min"
                )
            with limit_right:
                config["stellgroesse_max"] = st.number_input(
                    "Stellgröße max.", value=float(config["stellgroesse_max"]),
                    step=1.0, key="wirkplan_stellgroesse_max"
                )
            config["stellrate_max"] = st.number_input(
                "Maximale Stellrate [Einheit/s]", min_value=0.001,
                value=float(config["stellrate_max"]), step=1.0, key="wirkplan_stellrate_max"
            )

        with st.expander("4. Messkette", expanded=False):
            mess_left, mess_right = st.columns(2)
            with mess_left:
                config["messbereich_min"] = st.number_input(
                    "Messbereich min.", value=float(config["messbereich_min"]),
                    step=1.0, key="wirkplan_messbereich_min"
                )
                config["sensor_zeitkonstante_s"] = st.number_input(
                    "Sensor-Zeitkonstante [s]", min_value=0.0,
                    value=float(config["sensor_zeitkonstante_s"]), step=0.1,
                    key="wirkplan_sensor_zeitkonstante_s"
                )
            with mess_right:
                config["messbereich_max"] = st.number_input(
                    "Messbereich max.", value=float(config["messbereich_max"]),
                    step=1.0, key="wirkplan_messbereich_max"
                )
                config["messrauschen"] = st.number_input(
                    "Messrauschen (±)", min_value=0.0, value=float(config["messrauschen"]),
                    step=0.01, key="wirkplan_messrauschen"
                )

        with st.expander("5. Störeinflüsse", expanded=False):
            config["stoerungen_relevant"] = st.selectbox(
                "Gibt es relevante Störungen?",
                ["Ja", "Nein"],
                index=["Ja", "Nein"].index(config["stoerungen_relevant"]),
                key="wirkplan_stoerungen"
            )

            if config["stoerungen_relevant"] == "Ja":
                config["stoergroesse"] = st.text_input(
                    "Störgröße",
                    value=config["stoergroesse"],
                    key="wirkplan_stoergroesse"
                )
                config["stoerort"] = st.selectbox(
                    "Eingriffsort", ["Vor der Strecke", "Am Ausgang"],
                    index=["Vor der Strecke", "Am Ausgang"].index(config["stoerort"]),
                    key="wirkplan_stoerort"
                )
                config["stoerzeit_s"] = st.number_input(
                    "Störzeitpunkt [s]", min_value=0.0, value=float(config["stoerzeit_s"]),
                    step=0.5, key="wirkplan_stoerzeit_s"
                )
                config["stoerwert"] = st.number_input(
                    "Störsprung", value=float(config["stoerwert"]),
                    step=0.1, key="wirkplan_stoerwert"
                )
            else:
                config["stoergroesse"] = "keine relevante Störung"

        with st.expander("6. Regler- und Modell-Auslegung", expanded=False):
            config["auslegung"] = st.radio(
                "Auslegungsmodus", ["Automatisch", "Manuell"], horizontal=True,
                index=["Automatisch", "Manuell"].index(config["auslegung"]),
                key="wirkplan_auslegung"
            )
            if config["auslegung"] == "Manuell":
                config["man_controller_type"] = st.selectbox(
                    "Reglertyp", ["P", "PI", "PID"],
                    index=["P", "PI", "PID"].index(config["man_controller_type"]),
                    key="wirkplan_man_controller_type"
                )
                config["man_plant_type"] = st.selectbox(
                    "Streckentyp", ["PT1", "PT2"],
                    index=["PT1", "PT2"].index(config["man_plant_type"]),
                    key="wirkplan_man_plant_type"
                )
                for key, label, minimum, step in [
                    ("man_kp", "Kp", 0.0, 0.1), ("man_ki", "Ki", 0.0, 0.1),
                    ("man_kd", "Kd", 0.0, 0.1), ("man_ks", "Ks", 0.000001, 0.1),
                    ("man_ts", "Ts [s]", 0.000001, 0.1), ("man_zeta", "Dämpfungsgrad ζ", 0.01, 0.05),
                    ("man_omega0", "Eigenkreisfrequenz ω0 [rad/s]", 0.000001, 0.1),
                    ("man_setpoint", "Sollwert", -1000000.0, 0.1),
                    ("man_t_end", "Simulationsdauer [s]", 0.1, 1.0),
                    ("man_dt", "Zeitschritt dt [s]", 0.0001, 0.001),
                ]:
                    config[key] = st.number_input(
                        label, min_value=float(minimum), value=float(config[key]),
                        step=float(step), key=f"wirkplan_{key}"
                    )

        st.session_state.wirkplan_config = config
        validation_errors = validate_wirkplan_config(config)
        for validation_error in validation_errors:
            st.error(validation_error)

        derived = derive_controller_from_wirkplan(config)
        physical = derived.get("physical", {})

        st.divider()

        st.subheader("Abgeleiteter Regler")

        st.write(f"**Empfohlene Strecke:** {derived['plant_type']}")
        st.write(f"**Praxisstrategie:** {config['regelstrategie']}")
        st.write(f"**Simulationsregler:** {derived['controller_type']}")

        st.write("**Startparameter:**")
        st.write(f"- Kp = {derived['kp']}")
        st.write(f"- Ki = {derived['ki']}")
        st.write(f"- Kd = {derived['kd']}")
        st.write(f"- Ks = {derived['ks']}")
        st.write(f"- Ts = {derived['ts']}")
        st.write(f"- ζ = {derived['zeta']}")
        st.write(f"- ω0 = {derived['omega0']}")

        if st.button(
            "Parameter an alle Bereiche übertragen",
            width="stretch",
            disabled=bool(validation_errors),
            key="wirkplan_sync_all",
            help="Überträgt die aktuelle Ableitung in Simulation und visuellen Regelkreis-Builder, ohne die Seite zu wechseln.",
        ):
            apply_shared_parameters(
                shared_parameters_from_derived(derived),
                "dem physikalischen Wirkplan-Builder",
            )
            st.rerun()

        if st.button(
            "Wirkplan übernehmen und Simulation berechnen",
            type="primary",
            disabled=bool(validation_errors),
        ):
            apply_shared_parameters(
                shared_parameters_from_derived(derived),
                "dem physikalischen Wirkplan-Builder",
            )
            st.session_state.active_view = "simulation"
            st.rerun()

    with col_right:
        st.subheader("Grafischer Wirkplan")
        st.caption(
            f"{config['anlagenart']} › {config['regelungsvariante']} › "
            f"{config['stellglied_typ']}"
        )

        flow_state = build_wirkplan_flow(config)

        streamlit_flow(
            "wirkplan_flow",
            flow_state,
            fit_view=False,
            show_minimap=True,
            show_controls=True,
            allow_new_edges=False,
            animate_new_edges=False,
            height=500
        )

        derived = derive_controller_from_wirkplan(config)
        physical = derived.get("physical", {})

        if physical.get("active") and physical.get("supported"):
            st.subheader("Berechnete Anlagenkennwerte")
            metrics = physical.get("metrics", [])
            for start in range(0, len(metrics), 3):
                columns = st.columns(3)
                for column, (label, value) in zip(columns, metrics[start:start + 3]):
                    column.metric(label, value)

            if physical.get("model_note"):
                st.info(physical["model_note"])

            for warning in physical.get("warnings", []):
                st.warning(warning)

        st.subheader("Begründung der Ableitung")

        for begruendung in derived["begruendung"]:
            st.write("- " + begruendung)

        with st.expander("Aktueller Wirkplan-Datensatz"):
            st.json(config)

        with st.expander("Abgeleitete Simulationsdaten"):
            st.json(derived)

        with st.expander("Verfügbarer Praxiskatalog"):
            catalog_rows = []
            for category, variants in PRACTICAL_PROCESS_CATALOG.items():
                for variant_name, variant_profile in variants.items():
                    catalog_rows.append({
                        "Anlagenart": category,
                        "Regelungsvariante": variant_name,
                        "Grundmodell": variant_profile["model"],
                        "Stellglieder": ", ".join(variant_profile["actuators"]),
                        "Regelstrategien": ", ".join(variant_profile["strategies"]),
                    })
            st.dataframe(pd.DataFrame(catalog_rows), width="stretch", hide_index=True)
            st.caption(
                f"{len(PRACTICAL_PROCESS_CATALOG)} Anlagenarten mit "
                f"{len(catalog_rows)} Regelungsvarianten."
            )


# ------------------------------------------------------------
# Ansichten abfangen
# ------------------------------------------------------------

if st.session_state.active_view == "builder":
    render_visual_builder()
    st.stop()

if st.session_state.active_view == "wirkplan":
    render_wirkplan_builder()
    st.stop()


# ------------------------------------------------------------
# Oberfläche Simulation
# ------------------------------------------------------------

st.title("Regelkreis-Labor")

st.caption(
    "Interaktive Simulation eines geschlossenen Regelkreises mit Regler, "
    "Strecke, Rückführung und optionaler Störung."
)


# ------------------------------------------------------------
# Sidebar nur mit vier Optionsgruppen
# ------------------------------------------------------------

defaults = st.session_state.defaults
clear_pending_parameter_widgets()

with st.sidebar:

    if "parameter_sync_notice" in st.session_state:
        st.success(st.session_state.pop("parameter_sync_notice"))

    with st.expander("1. Regelkreis aufbauen", expanded=True):

        controller_type = st.selectbox(
            "Reglertyp",
            ["P", "PI", "PID"],
            index=["P", "PI", "PID"].index(st.session_state.controller_type),
            key="sim_controller_type",
        )

        plant_type = st.selectbox(
            "Streckentyp",
            ["PT1", "PT2"],
            index=["PT1", "PT2"].index(st.session_state.plant_type),
            key="sim_plant_type",
        )

        disturbance_position = st.selectbox(
            "Störung platzieren",
            ["Keine Störung", "Vor der Strecke", "Am Ausgang"],
            index=["Keine Störung", "Vor der Strecke", "Am Ausgang"].index(
                st.session_state.disturbance_position
            ),
            key="sim_disturbance_position",
        )

    with st.expander("2. Reglerparameter", expanded=False):

        kp = st.number_input(
            "Kp - Proportionalverstärkung",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults["kp"]),
            step=0.1,
            key="sim_kp",
            help="Kp bestimmt, wie stark der Regler direkt auf die aktuelle Regelabweichung reagiert."
        )

        if controller_type in ["PI", "PID"]:
            ki = st.number_input(
                "Ki - Integralverstärkung",
                min_value=0.0,
                max_value=100.0,
                value=float(defaults["ki"]),
                step=0.001,
                format="%.6f",
                key="sim_ki",
                help="Ki baut eine bleibende Regelabweichung über die Zeit ab."
            )
        else:
            ki = 0.0
            st.caption("Ki wird beim P-Regler automatisch auf 0 gesetzt.")

        if controller_type == "PID":
            kd = st.number_input(
                "Kd - Differentialverstärkung",
                min_value=0.0,
                max_value=100.0,
                value=float(defaults["kd"]),
                step=0.1,
                key="sim_kd",
                help="Kd reagiert auf schnelle Änderungen der Regelabweichung und kann Überschwingen dämpfen."
            )
        else:
            kd = 0.0
            st.caption("Kd ist nur beim PID-Regler relevant und wird automatisch auf 0 gesetzt.")

    with st.expander("3. Streckenparameter", expanded=False):

        ks = st.number_input(
            "Ks - Streckenverstärkung",
            min_value=0.000001,
            max_value=100.0,
            value=float(defaults["ks"]),
            step=0.01,
            format="%.6f",
            key="sim_ks",
            help="Ks beschreibt, wie stark die Strecke auf die Stellgröße reagiert."
        )

        if defaults.get("u_min") is not None and defaults.get("u_max") is not None:
            st.caption(
                f"Reales Stellglied aktiv: Stellgröße wird auf "
                f"{defaults['u_min']:.0f} bis {defaults['u_max']:.0f} % begrenzt."
            )

        if plant_type == "PT1":
            ts = st.number_input(
                "Ts - Zeitkonstante PT1 [s]",
                min_value=0.1,
                max_value=500000.0,
                value=float(defaults["ts"]),
                step=0.1,
                key="sim_ts",
                help="Ts beschreibt die Trägheit der PT1-Strecke."
            )

            zeta = defaults["zeta"]
            omega0 = defaults["omega0"]

            st.caption("ζ und ω0 sind für PT1 nicht relevant und werden automatisch intern gesetzt.")

        else:
            zeta = st.number_input(
                "Dämpfung ζ PT2",
                min_value=0.05,
                max_value=5.0,
                value=float(defaults["zeta"]),
                step=0.05,
                key="sim_zeta",
                help="ζ bestimmt, wie stark die PT2-Strecke schwingt oder gedämpft wird."
            )

            omega0 = st.number_input(
                "Eigenkreisfrequenz ω0 PT2 [rad/s]",
                min_value=0.1,
                max_value=100.0,
                value=float(defaults["omega0"]),
                step=0.1,
                key="sim_omega0",
                help="ω0 beschreibt die Eigenkreisfrequenz der PT2-Strecke."
            )

            ts = defaults["ts"]

            st.caption("Ts ist für PT2 nicht relevant und wird automatisch intern gesetzt.")

    with st.expander("4. Simulation", expanded=False):

        setpoint = st.number_input(
            "Sollwert w",
            value=float(defaults["setpoint"]),
            step=0.1,
            key="sim_setpoint",
            help="Der Sollwert ist die Führungsgröße, die die Regelgröße erreichen soll."
        )

        t_end = st.number_input(
            "Simulationsdauer [s]",
            min_value=1.0,
            max_value=1000000.0,
            value=float(defaults["t_end"]),
            step=1.0,
            key="sim_t_end",
            help="Legt fest, wie lange der zeitliche Verlauf berechnet und dargestellt wird.",
        )

        if st.session_state.get("schwierigkeitsgrad", "Fortgeschritten") == "Experte":
            dt = st.number_input(
                "Schrittweite dt [s]",
                min_value=0.001,
                max_value=60.0,
                value=float(defaults["dt"]),
                step=0.001,
                format="%.3f",
                key="sim_dt",
                help="Kleinere Schrittweiten erhöhen die Genauigkeit, benötigen aber mehr Rechenpunkte."
            )
        else:
            dt = defaults["dt"]
            st.caption(f"Schrittweite dt wird automatisch auf {dt} s gesetzt.")

        if disturbance_position != "Keine Störung":
            disturbance_time = st.number_input(
                "Störung ab Zeitpunkt [s]",
                min_value=0.0,
                max_value=float(max(200.0, t_end)),
                value=float(defaults["disturbance_time"]),
                step=0.5,
                key="sim_disturbance_time",
                help="Ab diesem Zeitpunkt wirkt der eingestellte Störsprung auf den Regelkreis.",
            )

            disturbance_value = st.number_input(
                "Störgröße d",
                value=float(defaults["disturbance_value"]),
                step=0.1,
                key="sim_disturbance_value",
            )
        else:
            disturbance_time = 0.0
            disturbance_value = 0.0
            st.caption("Keine Störung gewählt. Störzeitpunkt und Störgröße werden automatisch auf 0 gesetzt.")

    current_simulation_values = {
        "controller_type": controller_type,
        "plant_type": plant_type,
        "disturbance_position": disturbance_position,
        "kp": kp, "ki": ki, "kd": kd, "ks": ks, "ts": ts,
        "zeta": zeta, "omega0": omega0, "setpoint": setpoint,
        "t_end": t_end, "dt": dt,
        "disturbance_time": disturbance_time,
        "disturbance_value": disturbance_value,
        "u_min": defaults.get("u_min"), "u_max": defaults.get("u_max"),
    }

    st.divider()
    if st.button(
        "Parameter an beide Builder übertragen",
        type="primary",
        width="stretch",
        key="sim_sync_all",
        help="Überträgt den aktuellen Simulationsstand in den physikalischen und visuellen Builder.",
    ):
        apply_shared_parameters(current_simulation_values, "der Simulation")
        st.rerun()

    if st.button(
        "Parameter auf Standard zurücksetzen",
        width="stretch",
        key="sim_reset_parameters",
        help="Setzt Regler, Strecke, Sollwert, Störung und Simulationszeit auf die Programmstandards zurück.",
    ):
        reset_shared_parameters()
        st.rerun()


# ------------------------------------------------------------
# Plausibilitätsprüfung
# ------------------------------------------------------------

if dt >= t_end / 20:
    st.warning(
        "Die Schrittweite dt ist relativ groß. "
        "Für saubere Kurven sollte dt deutlich kleiner als die Simulationsdauer sein."
    )


# ------------------------------------------------------------
# Simulation ausführen
# ------------------------------------------------------------

df = simulate_control_loop(
    controller_type=controller_type,
    plant_type=plant_type,
    kp=kp,
    ki=ki,
    kd=kd,
    ks=ks,
    ts=ts,
    zeta=zeta,
    omega0=omega0,
    setpoint=setpoint,
    t_end=t_end,
    dt=dt,
    disturbance_position=disturbance_position,
    disturbance_time=disturbance_time,
    disturbance_value=disturbance_value,
    u_min=defaults.get("u_min"),
    u_max=defaults.get("u_max"),
)


# ------------------------------------------------------------
# Kennwerte berechnen
# ------------------------------------------------------------

final_value, steady_error, overshoot, settling_time = calculate_metrics(df, setpoint)


# ------------------------------------------------------------
# Kennzahlen anzeigen
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Endwert y", f"{final_value:.3f}")
col2.metric("bleibende Abweichung", f"{steady_error:.3f}")
col3.metric("Überschwingen", f"{overshoot:.1f} %")

if settling_time is None:
    col4.metric("Einschwingzeit", "nicht erreicht")
else:
    col4.metric("Einschwingzeit", f"{settling_time:.2f} s")


# ------------------------------------------------------------
# Automatische Auswertung
# ------------------------------------------------------------

st.subheader("Automatische Auswertung")

bewertung = []

if abs(steady_error) > 0.05:
    bewertung.append(
        "Es bleibt eine erkennbare Regelabweichung bestehen. "
        "Das ist typisch für einen reinen P-Regler oder eine zu schwache Reglerauslegung."
    )
else:
    bewertung.append(
        "Die bleibende Regelabweichung ist gering. Der Sollwert wird gut erreicht."
    )

if overshoot > 20:
    bewertung.append(
        "Das Überschwingen ist deutlich. Der Regelkreis ist relativ aggressiv eingestellt."
    )
elif overshoot > 5:
    bewertung.append(
        "Es ist ein moderates Überschwingen erkennbar."
    )
else:
    bewertung.append(
        "Das Überschwingen ist gering oder nicht vorhanden."
    )

if settling_time is None:
    bewertung.append(
        "Die Einschwingzeit wurde innerhalb der Simulationsdauer nicht erreicht. "
        "Die Simulationsdauer könnte zu kurz sein oder der Regelkreis schwingt zu stark."
    )
else:
    bewertung.append(
        f"Der Regelkreis erreicht das Toleranzband nach etwa {settling_time:.2f} s."
    )

if disturbance_position != "Keine Störung":
    bewertung.append(
        f"Die Störung wurde an der Stelle '{disturbance_position}' eingefügt. "
        "Im Zeitverlauf ist erkennbar, wie der Regler auf diese Störung reagiert."
    )

for text in bewertung:
    st.write("- " + text)


# ------------------------------------------------------------
# Automatisches Blockschaltbild
# ------------------------------------------------------------

st.subheader("Vervollständigter Regelkreis")

st.graphviz_chart(
    blockdiagramm(controller_type, plant_type, disturbance_position),
    width="stretch"
)


# ------------------------------------------------------------
# Zeitverlauf anzeigen
# ------------------------------------------------------------

st.subheader("Zeitverlauf")

fig, ax = plt.subplots(figsize=(10, 4.8))

ax.plot(
    df["Zeit [s]"],
    df["Sollwert w"],
    linestyle="--",
    label="Sollwert w"
)

ax.plot(
    df["Zeit [s]"],
    df["Regelgröße y"],
    linewidth=2,
    label="Regelgröße y"
)

ax.plot(
    df["Zeit [s]"],
    df["Stellgröße u"],
    alpha=0.8,
    label="Stellgröße u"
)

if disturbance_position != "Keine Störung":
    ax.plot(
        df["Zeit [s]"],
        df["Störung d"],
        linestyle=":",
        label="Störung d"
    )

ax.set_xlabel("Zeit [s]")
ax.set_ylabel("Signalwert")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="best")

st.pyplot(fig)


# ------------------------------------------------------------
# Rohdaten anzeigen
# ------------------------------------------------------------

with st.expander("Rohdaten anzeigen"):
    st.dataframe(df, width="stretch")


# ------------------------------------------------------------
# Technische Einordnung
# ------------------------------------------------------------

with st.expander("Technische Einordnung"):
    st.markdown(
        f"""
        **Aktueller Aufbau:** {controller_type}-Regler mit {plant_type}-Strecke.

        **Reglerprinzip:**  
        Die Regeldifferenz wird berechnet aus:

        $$
        e(t) = w(t) - y(t)
        $$

        Daraus bildet der Regler die Stellgröße:

        $$
        u(t) = K_p \\cdot e(t) + K_i \\int e(t)dt + K_d \\frac{{de(t)}}{{dt}}
        $$

        Je nach Auswahl werden P-, I- und D-Anteil aktiviert oder deaktiviert.

        **Rückführung:**  
        Die Regelgröße y(t) wird auf den Eingang zurückgeführt und vom Sollwert w(t) abgezogen.

        **Störung:**  
        {disturbance_position}
        """
    )

