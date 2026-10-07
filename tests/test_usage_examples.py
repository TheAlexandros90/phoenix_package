import unittest

import pandas as pd

from phoenix import Phoenix


class PhoenixUsageExamples(unittest.TestCase):
    def test_cleaning_workflow_example(self):
        df = pd.DataFrame(
            {
                " Nombre Cliente ": [" ana ", "LUIS", None],
                "Activo Texto": ["si", "no", "verdadero"],
                "Ventas": ["100", "250", "error"],
                "Observación": ["  alta prioridad", "sin incidencia  ", None],
            }
        )

        phoenix = Phoenix(df.copy())
        phoenix.estandarizar_nombres_columnas()
        phoenix.normalizar_texto(["nombre_cliente", "observacion"], case="title")
        phoenix.convertir_booleanos_texto(["activo_texto"], allow_new_nulls=True)
        phoenix.convertir_a_numerico(["ventas"], allow_new_nulls=True)

        self.assertEqual(
            list(phoenix.df.columns),
            ["nombre_cliente", "activo_texto", "ventas", "observacion"],
        )
        self.assertEqual(phoenix.df["nombre_cliente"].tolist(), ["Ana", "Luis", None])
        self.assertEqual(phoenix.df["activo_texto"].tolist(), [True, False, True])
        self.assertEqual(phoenix.df["ventas"].tolist()[0:2], [100.0, 250.0])
        self.assertTrue(pd.isna(phoenix.df["ventas"].iloc[2]))

    def test_quality_report_example(self):
        df = pd.DataFrame(
            {
                "cliente": ["Ana", "Ana", "Luis"],
                "segmento": ["retail", "retail", "empresa"],
                "ventas": [100, 100, 300],
                "vacia": [None, None, None],
            }
        )

        phoenix = Phoenix(df)
        report = phoenix.reporte_calidad_datos(top_n=5)

        self.assertEqual(int(report["duplicates"].iloc[0]["duplicate_rows"]), 1)
        self.assertEqual(report["all_null_columns"]["column"].tolist(), ["vacia"])
        self.assertIn("duplicate_rows", report["suggestions"]["issue"].tolist())


if __name__ == "__main__":
    unittest.main()