import unittest

import pandas as pd

from phoenix import Phoenix


class PhoenixSmokeTests(unittest.TestCase):
    def test_resumen_nulos_reporta_columnas_con_nulos(self):
        df = pd.DataFrame({"a": [1, None], "b": ["x", "y"]})

        phoenix = Phoenix(df)
        resumen = phoenix.resumen_nulos()

        self.assertEqual(int(resumen.loc["a", "null_count"]), 1)
        self.assertNotIn("b", resumen.index)

    def test_convertir_booleanos_texto_convierte_valores_validos(self):
        df = pd.DataFrame({"activo": ["si", "no", "verdadero", None]})

        phoenix = Phoenix(df)
        phoenix.convertir_booleanos_texto(["activo"], allow_new_nulls=True)

        self.assertEqual(str(phoenix.df["activo"].dtype), "boolean")
        self.assertEqual(phoenix.df["activo"].tolist(), [True, False, True, pd.NA])

    def test_estandarizar_nombres_columnas_normaliza(self):
        df = pd.DataFrame({" Nombre Cliente ": ["Ana"], "Ciudad / Zona": ["Madrid"]})

        phoenix = Phoenix(df)
        phoenix.estandarizar_nombres_columnas()

        self.assertEqual(list(phoenix.df.columns), ["nombre_cliente", "ciudad_zona"])


if __name__ == "__main__":
    unittest.main()