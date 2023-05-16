from typing import List, Tuple


VSA_VOZLISCA = {}


class Vozlisce:
    def __init__(
        self,
        tip: str,
        opis: str,
        stars: 'Vozlisce',
        otroci: List['Vozlisce']
    ):
        self.tip = tip
        self.opis = opis
        self.stars = stars
        self.otroci = otroci

    def __str__(self):
        return self._str_pomo("")

    def _str_pomo(self, presledek):
        prva_vrsta = f"{presledek}Vozlice(tip='{self.tip}'; opis='{self.opis}'; otroci=["
        if not self.otroci:
            return f"{prva_vrsta}])"
        otroci = ",\n".join(
            otrok._str_pomo(presledek + "  ") for otrok in self.otroci
        )
        return f"{prva_vrsta}\n{otroci}\n{presledek}])"

    @staticmethod
    def pretvori_iz_agde(agda_vozlisce, has_parent=False):
        i = agda_vozlisce._id
        if i in VSA_VOZLISCA:
            return VSA_VOZLISCA[i]
        v = Vozlisce(
            agda_vozlisce.node_type.value,
            agda_vozlisce.node_description,
            None,
            []
        )
        VSA_VOZLISCA[i] = v
        otroci = [
            Vozlisce.pretvori_iz_agde(otrok, True)
            for otrok in agda_vozlisce.children
        ]
        v.otroci = otroci
        s = agda_vozlisce.parent
        if has_parent:
            v.stars = VSA_VOZLISCA[s._id]
        return v


class Definicija:
    def __init__(
        self,
        koren: Vozlisce,
        ime: Tuple[str, str, str],
        tip: Vozlisce,
        telo: Vozlisce
    ):
        self.koren = koren
        self.ime = ime
        self.tip = tip
        self.telo = telo

    @staticmethod
    def pretvori_iz_agde(agda_definition):
        koren = Vozlisce.pretvori_iz_agde(agda_definition.root)
        return Definicija(
            koren,
            agda_definition.name,
            koren.otroci[1],
            koren.otroci[2]
        )

    def __str__(self):
        return str(self.koren)
