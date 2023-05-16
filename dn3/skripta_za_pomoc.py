from typing import List
import re
import networkx as nx
import pickle
from def_drevesa import Definicija, Vozlisce


def nalozi_podatke():
    """
    Pokliče vse ostalo in naloži
    - graf: nx.MultiDiGraph
    - učne in testne definicije: {ime: Definicija}, kjer je ime
      (str, str, str)
    - testne povezave: [p1, p2, ...], kjer je
      p = [ime, ime, tip, 1/0]
    """
    graf_dato = "graf.txt"
    def_ucna_dato = "def_ucna.pickle"
    def_testna_dato = "def_testna.pickle"
    testne_povezave = "testne_povezave.csv"
    graf = nalozi_graf(graf_dato)
    def_ucna = nalozi_def(def_ucna_dato)
    def_testna = nalozi_def(def_testna_dato)
    povezave = nalozi_povezave(testne_povezave)
    return graf, def_ucna, def_testna, povezave


def nalozi_graf(graf_dato):
    graf = nx.MultiDiGraph()
    with open(graf_dato, encoding="utf-8") as f:
        for vrsta in f:
            deli = vrsta.split(";")[1:]
            deli = [eval(d) for d in deli]
            if vrsta.startswith("V"):
                vozlisce, lastnosti = deli
                graf.add_node(vozlisce, **lastnosti)
            else:
                od, do, tip, lastnosti = deli
                graf.add_edge(od, do, tip, **lastnosti)
    return graf


def nalozi_def(def_dato):
    with open(def_dato, "rb") as f:
        return pickle.load(f)


def nalozi_povezave(povezave_dato):
    povezave = []
    with open(povezave_dato, encoding="utf-8") as f:
        f.readline()  # glava
        for vrsta in f:
            od, do, tip, prisotnost = [
                eval(d) for d in vrsta.split(";")
            ]
            povezave.append((od, do, tip, prisotnost))
    return povezave


def nalozi_vektorje_besed():
    """
    Vrne slovar {beseda: vektor, ...}
    """
    vektorji = {}
    with open("besedisce.txt", encoding="utf-8") as f:
        n_vektorji, dim = map(int, f.readline().split(" "))
        for vrsta in f:
            kosi = vrsta.split(" ")
            beseda = kosi[0]
            vektor = [float(kos) for kos in kosi[1:]]
            vektorji[beseda] = vektor
    assert len(vektorji) == n_vektorji, (len(vektorji), n_vektorji)
    assert {len(v) for v in vektorji.values()} == {dim}
    return vektorji


def razbij_niz(niz: str) -> List[str]:
    """
    Razbije niz na 'osnovne delce', kot jih definirajo
    kamelje grbe, pomišljaji, ne-ascii znaki, števila ...

    :param name: niz, npr. ``to₂₂Je_paZelo-dolgoIme_ani12∘21``

    :return: a list of parts, for example
        ``['to', '₂₂', 'je', 'pa', 'zelo', 'dolgo', 'ime', 'ani', '12', '∘', '21']``

    """
    def naj_razbijem(i: int):
        """
        Ali naj razbijemo pred i-tim znakom?
        :param i:
        :return:
        """
        if i == 0:
            return False
        ta = niz[i]
        prej = niz[i - 1]
        if ta.isnumeric() and prej.isnumeric():
            # dve števki
            return False
        elif ta.isalpha() and prej.isalpha() and \
                not (ta.upper() == ta and prej.lower() == prej):
            # dve ne-kamelji črki
            return False
        else:
            return True

    posebni = "_-."
    razsirjeno = []
    for polozaj, znak in enumerate(niz):
        if naj_razbijem(polozaj):
            razsirjeno.append("_")
        if znak not in posebni:
            razsirjeno.append(znak)
    return re.sub("_+", "_", "".join(razsirjeno)).lower().split("_")


poglej_podatke = True
poglej_vektorje = True
if poglej_podatke:
    # tale korak morda traja nekaj časa
    G, DEF_UCNE, DEF_TESTNE, TESTNE_POVEZAVE = nalozi_podatke()
    ########
    # GRAF #
    ########

    print(
        "Dimenzije G = (V, E): (|V|, |E|) = "
        f"({len(G.nodes)}, {len(G.edges)})"
    )

    # kratka analiza vozlišč: njihov tip se skriva v lastnosti "label"
    vrste_vozlisc = sorted(set(G.nodes[voz]["label"] for voz in G))
    print(f"Vrste vozlišč v grafu: {vrste_vozlisc}")
    # kratka analiza povezav
    vrste_lastnosti = {}
    for u, v, tip_povezave, lastnosti in G.edges(keys=True, data=True):
        if tip_povezave not in vrste_lastnosti:
            vrste_lastnosti[tip_povezave] = set(lastnosti)
        else:
            # preverimo, da imajo vse istega tipa iste lastnosti
            assert set(lastnosti) == vrste_lastnosti[tip_povezave]
    print("Vrste povezav v grafu:")
    for par in vrste_lastnosti.items():
        print("   ", par)

    ##############
    # DEFINICIJE #
    ##############

    # ucne in testne definicije imajo enako strukturo, testne
    # so bile oskubljene

    # izberimo eno in jo izpišimo: imena defincij so trodelna
    # zadnja komponenta pove, kje definicijo najdemo
    # oglejmo si vgrajeno def. naravnih števil
    kljuc = ('13537827747504913145', '6', 'Agda.Builtin.Nat.Nat')
    # kljuc_trantizivnost:
    # ('5359185163178559078', '70', 'Relation.Binary.Definitions.Transitive')
    izbrana_def: Definicija = DEF_UCNE[kljuc]
    print(f"Tako je videti definicija kot drevo:\n{izbrana_def}")

    # tudi vozlišča v definiciji so različnih vrst:
    # preglejmo samo izbrano


    def sprehod_po_definiciji(koren: Vozlisce):
        yield koren
        for otrok in koren.otroci:
            yield from sprehod_po_definiciji(otrok)


    print("Vrste vozlišč:")
    print(f"    koren: {izbrana_def.koren.tip}")
    print(f"    koren.otroci[0]: {izbrana_def.koren.otroci[0].tip}")
    print(f"    koren.otroci[1]: {izbrana_def.koren.otroci[1].tip}")
    vse_vrste = {voz.tip for voz in sprehod_po_definiciji(izbrana_def.koren)}
    print(f"    Vse vrste iz izbrane definicije: {sorted(vse_vrste)}")


    ###################
    # TESTNE POVEZAVE #
    ###################

    print("Nekaj testnih povezav:")
    for povezava in TESTNE_POVEZAVE[:3] + TESTNE_POVEZAVE[-3:]:
        print("   ", povezava)

if poglej_vektorje:
    ##################
    # VEKTORJI BESED #
    ##################
    print("Nekaj besedišča:")
    VEKTORJI_BESED = nalozi_vektorje_besed()
    n = 5
    for beseda, vektor in VEKTORJI_BESED.items():
        print("   ", beseda, f"[{str(vektor[:5])[1:-1]}, ...]")
        n -= 1
        if n == 0:
            break
