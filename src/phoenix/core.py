"""Phoenix: utilities for cleaning and profiling pandas DataFrames."""

import logging
import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)



class Phoenix:

    

    def __init__(self, df):

        """

        Inicializa la clase con un DataFrame.

        """

        if df.empty:

            raise ValueError("El DataFrame estÃ¡ vacio")

        self.data = df

        self.df = df

        self.last_reduce_mem_usage_report = []

        self.last_outlier_report = {}

        self.last_variable_classification_report = {}

        self.last_dataset_report = {}

        self.last_type_conversion_report = []

        self.last_data_quality_report = {}

        self.last_column_standardization_report = {}

        self.last_group_summary = None



    def _sync_dataframes(self):

        self.df = self.data

        return self.data



    def _get_frame(self, frame_name='df'):

        if frame_name not in {'df', 'data'}:

            raise ValueError("frame_name must be 'df' or 'data'")

        return getattr(self, frame_name)



    def _ensure_column_exists(self, columna, frame_name='df'):

        frame = self._get_frame(frame_name)

        if columna not in frame.columns:

            raise ValueError(f"{columna} does not exist in the DataFrame")

        return columna



    def _ensure_columns_exist(self, columns, frame_name='data'):

        frame = self._get_frame(frame_name)

        missing_cols = [col for col in columns if col not in frame.columns]

        if missing_cols:

            raise ValueError(f"{missing_cols} do not exist in the DataFrame")

        return list(columns)



    def _serie_como_texto(self, columna):

        self._ensure_column_exists(columna, frame_name='df')

        return self.df[columna].apply(lambda value: None if pd.isna(value) else str(value))



    def _split_safe(self, value, separator=','):

        if value is None:

            return []

        return [parte.strip() for parte in str(value).split(separator)]



    def _obtener_parte(self, value, index, separator=','):

        partes = self._split_safe(value, separator=separator)

        if not partes:

            return None

        try:

            return partes[index]

        except IndexError:

            return None



    def _sync_text_result(self, nueva_columna, serie_resultado):

        self.df[nueva_columna] = serie_resultado

        self.data = self.df

        return self._sync_dataframes()



    def _numeric_series(self, columna):

        self._ensure_column_exists(columna, frame_name='df')

        serie = pd.to_numeric(self.df[columna], errors='coerce').dropna()

        if serie.empty:

            raise ValueError(f"{columna} has no valid numeric values")

        return serie



    def _make_hashable(self, value):

        if isinstance(value, list):

            return tuple(self._make_hashable(item) for item in value)

        if isinstance(value, tuple):

            return tuple(self._make_hashable(item) for item in value)

        if isinstance(value, set):

            return tuple(sorted(self._make_hashable(item) for item in value))

        if isinstance(value, dict):

            return tuple(sorted((key, self._make_hashable(item)) for key, item in value.items()))

        if isinstance(value, np.ndarray):

            return tuple(self._make_hashable(item) for item in value.tolist())

        return value



    def _safe_nunique(self, serie, dropna=True):

        if dropna:

            serie = serie.dropna()

        try:

            return int(serie.nunique(dropna=dropna))

        except TypeError:

            return int(serie.apply(self._make_hashable).nunique(dropna=dropna))



    def _registrar_outliers(self, metodo, columna, outliers, show_plot=False):

        self.last_outlier_report = {

            'method': metodo,

            'column': columna,

            'count': len(outliers),

            'outliers': outliers

        }

        logger.info("Outliers detected with %s in %s: %s", metodo, columna, len(outliers))

        if show_plot and outliers:

            self.grafico_interactivo(columna, outliers)

        return outliers



    def resumen_nulos(self, sort=True):

        """Devuelve un resumen de valores nulos por columna."""

        resumen = pd.DataFrame({

            'null_count': self.df.isnull().sum(),

            'null_percentage': (self.df.isnull().mean() * 100).round(2)

        })

        resumen = resumen[resumen['null_count'] > 0]

        if sort:

            resumen = resumen.sort_values(by='null_count', ascending=False)

        return resumen



    def resumen_memoria(self):

        """Devuelve un resumen de memoria por columna."""

        memoria = self.df.memory_usage(deep=True)

        resumen = pd.DataFrame({

            'memory_bytes': memoria,

            'memory_mb': (memoria / 1024**2).round(4),

            'dtype': [self.df.index.dtype] + [self.df[col].dtype for col in self.df.columns]

        }, index=['Index'] + self.df.columns.tolist())

        return resumen



    def resumen_dataset(self, top_n=10, include_sample=False):

        """Devuelve un reporte general del dataset con mÃ©tricas clave."""

        if top_n < 1:

            raise ValueError("top_n must be greater than or equal to 1")



        total_filas, total_columnas = self.df.shape

        memoria_total_mb = round(self.df.memory_usage(deep=True).sum() / 1024**2, 4)

        overview = pd.DataFrame([

            {

                'rows': total_filas,

                'columns': total_columnas,

                'duplicate_rows': int(self.df.duplicated().sum()),

                'total_nulls': int(self.df.isnull().sum().sum()),

                'total_memory_mb': memoria_total_mb

            }

        ])



        nulls = self.resumen_nulos(sort=True)

        memory = self.resumen_memoria().drop(index='Index', errors='ignore')

        memory = memory.sort_values(by='memory_bytes', ascending=False).head(top_n)

        dtypes = self.df.dtypes.astype(str).value_counts().rename_axis('dtype').reset_index(name='count')

        unique_count = pd.Series({

            column: self._safe_nunique(self.df[column], dropna=True)

            for column in self.df.columns

        })

        cardinality = pd.DataFrame({

            'unique_count': unique_count,

            'unique_percentage': ((unique_count / max(total_filas, 1)) * 100).round(2)

        }).sort_values(by='unique_count', ascending=False).head(top_n)



        report = {

            'overview': overview,

            'nulls': nulls,

            'memory': memory,

            'dtypes': dtypes,

            'cardinality': cardinality

        }



        if include_sample:

            report['sample'] = self.df.head(top_n).copy()



        self.last_dataset_report = report

        return report



    def eliminar_duplicados(self, subset=None, keep='first'):

        """Elimina filas duplicadas del DataFrame."""

        if subset is not None:

            subset = self._ensure_columns_exist(subset, frame_name='data')

        self.data = self.data.drop_duplicates(subset=subset, keep=keep)

        return self._sync_dataframes()



    def normalizar_texto(self, columns, case='lower', strip=True, remove_extra_spaces=True):

        """Normaliza texto en columnas seleccionadas."""

        columns = self._ensure_columns_exist(columns, frame_name='data')



        def normalize_value(value):

            if pd.isna(value):

                return value

            text = str(value)

            if strip:

                text = text.strip()

            if remove_extra_spaces:

                text = re.sub(r'\s+', ' ', text)

            if case == 'lower':

                text = text.lower()

            elif case == 'upper':

                text = text.upper()

            elif case == 'title':

                text = text.title()

            elif case not in {'keep', None}:

                raise ValueError("case must be one of: 'lower', 'upper', 'title', 'keep'")

            return text



        for column in columns:

            self.data[column] = self.data[column].apply(normalize_value)



        return self._sync_dataframes()



    def estandarizar_nombres_columnas(self, lowercase=True, strip=True, remove_accents=True, replace_spaces='_', remove_special_chars=True, deduplicate=True):

        """Estandariza nombres de columnas para facilitar trabajo posterior."""

        original_columns = list(self.data.columns)

        normalized_columns = []

        used_names = set()



        for column in original_columns:

            new_name = str(column)

            if strip:

                new_name = new_name.strip()

            if remove_accents:

                new_name = unicodedata.normalize('NFKD', new_name).encode('ascii', 'ignore').decode('ascii')

            if lowercase:

                new_name = new_name.lower()

            if replace_spaces is not None:

                new_name = re.sub(r'\s+', replace_spaces, new_name)

            if remove_special_chars:

                allowed_separator = re.escape(replace_spaces) if replace_spaces else ''

                new_name = re.sub(rf'[^0-9a-zA-Z{allowed_separator}]', '', new_name)

            if replace_spaces:

                separator_pattern = re.escape(replace_spaces)

                new_name = re.sub(rf'{separator_pattern}+', replace_spaces, new_name).strip(replace_spaces)

            if not new_name:

                new_name = 'columna'



            if deduplicate:

                base_name = new_name

                suffix = 2

                while new_name in used_names:

                    new_name = f'{base_name}_{suffix}'

                    suffix += 1



            used_names.add(new_name)

            normalized_columns.append(new_name)



        mapping = dict(zip(original_columns, normalized_columns))

        self.data = self.data.rename(columns=mapping)

        self.last_column_standardization_report = mapping

        return self._sync_dataframes()



    def convertir_booleanos_texto(self, columns, true_values=None, false_values=None, allow_new_nulls=False):

        """Convierte columnas de texto a booleano nullable de forma segura."""

        columns = self._ensure_columns_exist(columns, frame_name='data')

        true_values = true_values or {'si', 'sÃ­', 'true', 'verdadero', '1', 'yes', 'y', 't'}

        false_values = false_values or {'no', 'false', 'falso', '0', 'off', 'n', 'f'}

        true_values = {str(value).strip().lower() for value in true_values}

        false_values = {str(value).strip().lower() for value in false_values}



        report = []

        for column in columns:

            original = self.data[column]

            original_nulls = int(original.isna().sum())

            counter = {'unmapped': 0}



            def convert_value(value):

                if pd.isna(value):

                    return pd.NA

                if isinstance(value, bool):

                    return value

                token = str(value).strip().lower()

                if token in true_values:

                    return True

                if token in false_values:

                    return False

                counter['unmapped'] += 1

                return pd.NA



            converted = original.apply(convert_value).astype('boolean')

            converted_nulls = int(converted.isna().sum())

            added_nulls = max(converted_nulls - original_nulls, 0)



            if not allow_new_nulls and added_nulls > 0:

                report.append({

                    'column': column,

                    'dtype_original': str(original.dtype),

                    'dtype_final': str(original.dtype),

                    'status': 'skipped',

                    'detail': f'unsafe_boolean_conversion_added_{added_nulls}_nulls'

                })

                continue



            self.data[column] = converted

            report.append({

                'column': column,

                'dtype_original': str(original.dtype),

                'dtype_final': str(converted.dtype),

                'status': 'converted',

                'detail': f'new_nulls_added={added_nulls}; unmapped_values={counter["unmapped"]}'

            })



        self.last_type_conversion_report = report

        return self._sync_dataframes()



    def convertir_a_numerico(self, columns, errors='coerce', downcast=None, allow_new_nulls=False):

        """Convierte columnas a tipo numÃ©rico sin introducir nulos nuevos salvo que se permita."""

        columns = self._ensure_columns_exist(columns, frame_name='data')

        valid_downcasts = {None, 'integer', 'signed', 'unsigned', 'float'}

        if downcast not in valid_downcasts:

            raise ValueError("downcast must be one of: None, 'integer', 'signed', 'unsigned', 'float'")



        report = []

        for column in columns:

            original = self.data[column]

            original_nulls = int(original.isna().sum())

            converted = pd.to_numeric(original, errors=errors, downcast=downcast)

            converted_nulls = int(converted.isna().sum())

            added_nulls = max(converted_nulls - original_nulls, 0)



            if not allow_new_nulls and added_nulls > 0:

                report.append({

                    'column': column,

                    'dtype_original': str(original.dtype),

                    'dtype_final': str(original.dtype),

                    'status': 'skipped',

                    'detail': f'unsafe_conversion_added_{added_nulls}_nulls'

                })

                continue



            self.data[column] = converted

            report.append({

                'column': column,

                'dtype_original': str(original.dtype),

                'dtype_final': str(converted.dtype),

                'status': 'converted',

                'detail': f'new_nulls_added={added_nulls}'

            })



        self.last_type_conversion_report = report

        return self._sync_dataframes()



    def convertir_a_datetime(self, columns, errors='coerce', format=None, dayfirst=False, yearfirst=False, utc=False, allow_new_nulls=False):

        """Convierte columnas a datetime sin introducir nulos nuevos salvo que se permita."""

        columns = self._ensure_columns_exist(columns, frame_name='data')



        report = []

        for column in columns:

            original = self.data[column]

            original_nulls = int(original.isna().sum())

            converted = pd.to_datetime(

                original,

                errors=errors,

                format=format,

                dayfirst=dayfirst,

                yearfirst=yearfirst,

                utc=utc

            )

            converted_nulls = int(converted.isna().sum())

            added_nulls = max(converted_nulls - original_nulls, 0)



            if not allow_new_nulls and added_nulls > 0:

                report.append({

                    'column': column,

                    'dtype_original': str(original.dtype),

                    'dtype_final': str(original.dtype),

                    'status': 'skipped',

                    'detail': f'unsafe_datetime_conversion_added_{added_nulls}_nulls'

                })

                continue



            self.data[column] = converted

            report.append({

                'column': column,

                'dtype_original': str(original.dtype),

                'dtype_final': str(converted.dtype),

                'status': 'converted',

                'detail': f'new_nulls_added={added_nulls}'

            })



        self.last_type_conversion_report = report

        return self._sync_dataframes()



    def convertir_a_categoria(self, columns, max_unique_ratio=0.5, allow_high_cardinality=False):

        """Convierte columnas a category solo si la cardinalidad y la memoria lo hacen seguro."""

        columns = self._ensure_columns_exist(columns, frame_name='data')

        if not 0 < max_unique_ratio <= 1:

            raise ValueError("max_unique_ratio must be between 0 and 1")



        report = []

        total_rows = max(len(self.data), 1)



        for column in columns:

            original = self.data[column]

            dtype_original = original.dtype

            unique_count = self._safe_nunique(original, dropna=True)

            unique_ratio = unique_count / total_rows



            if isinstance(dtype_original, pd.CategoricalDtype):

                report.append({

                    'column': column,

                    'dtype_original': str(dtype_original),

                    'dtype_final': str(dtype_original),

                    'status': 'skipped',

                    'detail': 'already_category'

                })

                continue



            if pd.api.types.is_datetime64_any_dtype(dtype_original):

                report.append({

                    'column': column,

                    'dtype_original': str(dtype_original),

                    'dtype_final': str(dtype_original),

                    'status': 'skipped',

                    'detail': 'datetime_not_supported'

                })

                continue



            if not allow_high_cardinality and unique_ratio > max_unique_ratio:

                report.append({

                    'column': column,

                    'dtype_original': str(dtype_original),

                    'dtype_final': str(dtype_original),

                    'status': 'skipped',

                    'detail': f'high_cardinality_ratio={unique_ratio:.4f}'

                })

                continue



            converted = original.astype('category')

            original_memory = original.memory_usage(deep=True)

            converted_memory = converted.memory_usage(deep=True)



            if converted_memory >= original_memory:

                report.append({

                    'column': column,

                    'dtype_original': str(dtype_original),

                    'dtype_final': str(dtype_original),

                    'status': 'skipped',

                    'detail': 'no_memory_gain'

                })

                continue



            self.data[column] = converted

            report.append({

                'column': column,

                'dtype_original': str(dtype_original),

                'dtype_final': str(converted.dtype),

                'status': 'converted',

                'detail': f'unique_count={unique_count}; unique_ratio={unique_ratio:.4f}'

            })



        self.last_type_conversion_report = report

        return self._sync_dataframes()



    def reporte_calidad_datos(self, top_n=10, include_sample=False, high_cardinality_threshold=0.9):

        """Devuelve un reporte de calidad con problemas frecuentes y sugerencias."""

        if top_n < 1:

            raise ValueError("top_n must be greater than or equal to 1")

        if not 0 < high_cardinality_threshold <= 1:

            raise ValueError("high_cardinality_threshold must be between 0 and 1")



        total_rows = max(len(self.df), 1)

        base_report = self.resumen_dataset(top_n=top_n, include_sample=include_sample)



        duplicate_rows = self.df.duplicated()

        duplicate_summary = pd.DataFrame([

            {

                'duplicate_rows': int(duplicate_rows.sum()),

                'duplicate_percentage': round(float(duplicate_rows.mean() * 100), 2)

            }

        ])



        all_null_mask = self.df.isnull().all()

        all_null_column_names = self.df.columns[all_null_mask].tolist()

        all_null_columns = pd.DataFrame({

            'column': all_null_column_names,

            'null_percentage': [100.0] * len(all_null_column_names)

        })



        nunique_including_nulls = pd.Series({

            column: self._safe_nunique(self.df[column], dropna=False)

            for column in self.df.columns

        })

        constant_column_names = [

            column for column in nunique_including_nulls[nunique_including_nulls <= 1].index

            if column not in all_null_column_names

        ]

        constant_columns = pd.DataFrame({

            'column': constant_column_names,

            'unique_count': [int(nunique_including_nulls[column]) for column in constant_column_names],

            'example_value': [self.df[column].iloc[0] if len(self.df) else None for column in constant_column_names]

        })



        unique_count = pd.Series({

            column: self._safe_nunique(self.df[column], dropna=True)

            for column in self.df.columns

        })

        unique_ratio = (unique_count / total_rows).round(4)

        high_cardinality = pd.DataFrame({

            'unique_count': unique_count,

            'unique_ratio': unique_ratio

        })

        high_cardinality = high_cardinality[high_cardinality['unique_ratio'] >= high_cardinality_threshold]

        high_cardinality = high_cardinality.sort_values(by='unique_ratio', ascending=False).head(top_n)



        text_issue_rows = []

        text_columns = self.df.select_dtypes(include=['object', 'string']).columns

        for column in text_columns:

            serie = self.df[column].dropna().astype(str)

            if serie.empty:

                continue

            leading_or_trailing = int((serie != serie.str.strip()).sum())

            extra_spaces = int(serie.str.contains(r'\s{2,}', regex=True).sum())

            case_inconsistency = int(serie.str.lower().nunique() < serie.nunique())

            if leading_or_trailing or extra_spaces or case_inconsistency:

                text_issue_rows.append({

                    'column': column,

                    'leading_or_trailing_spaces': leading_or_trailing,

                    'extra_spaces': extra_spaces,

                    'possible_case_inconsistency': case_inconsistency

                })

        text_issues = pd.DataFrame(text_issue_rows)



        suggestions = []

        if int(duplicate_rows.sum()) > 0:

            suggestions.append({'issue': 'duplicate_rows', 'recommendation': 'Usar eliminar_duplicados() para limpiar filas repetidas.'})

        for column in all_null_column_names:

            suggestions.append({'issue': f'all_null_column:{column}', 'recommendation': f'Revisar o eliminar la columna {column} porque estÃ¡ completamente vacÃ­a.'})

        for column in constant_column_names:

            suggestions.append({'issue': f'constant_column:{column}', 'recommendation': f'Revisar si la columna {column} aporta informaciÃ³n; parece constante.'})

        for column in high_cardinality.index.tolist():

            suggestions.append({'issue': f'high_cardinality:{column}', 'recommendation': f'La columna {column} tiene alta cardinalidad; evitar category salvo que sea un identificador controlado.'})

        for row in text_issue_rows:

            suggestions.append({'issue': f'text_normalization:{row["column"]}', 'recommendation': f'Normalizar la columna {row["column"]} con normalizar_texto().'})



        suggestions_df = pd.DataFrame(suggestions)



        report = {

            **base_report,

            'duplicates': duplicate_summary,

            'constant_columns': constant_columns,

            'all_null_columns': all_null_columns,

            'high_cardinality_columns': high_cardinality,

            'text_issues': text_issues,

            'suggestions': suggestions_df

        }



        self.last_data_quality_report = report

        return report



    def resumen_por_grupo(self, group_by, agg_columns=None, metrics=None, dropna=False, sort_by=None, ascending=False):

        """Resume mÃ©tricas por grupo para auditorÃ­a rÃ¡pida del dataset."""

        group_columns = [group_by] if isinstance(group_by, str) else list(group_by)

        group_columns = self._ensure_columns_exist(group_columns, frame_name='df')



        if agg_columns is None:

            agg_columns = [

                column for column in self.df.select_dtypes(include=[np.number]).columns

                if column not in group_columns

            ]

        else:

            agg_columns = self._ensure_columns_exist(agg_columns, frame_name='df')



        metrics = metrics or ['count', 'mean', 'median', 'sum']

        grouped = self.df.groupby(group_columns, dropna=dropna, observed=False)



        if agg_columns:

            summary = grouped[agg_columns].agg(metrics)

            if isinstance(summary.columns, pd.MultiIndex):

                summary.columns = [f'{column}_{metric}' for column, metric in summary.columns]

        else:

            summary = grouped.size().to_frame('count')



        summary = summary.reset_index()

        if sort_by is not None and sort_by in summary.columns:

            summary = summary.sort_values(by=sort_by, ascending=ascending)



        self.last_group_summary = summary

        return summary



    # Valores Nulos



    def eliminar_filas_valores_faltantes_en_columnas(self, columns):

        """Eliminar las filas con valores faltantes en columnas especÃ­ficas."""

        columns = self._ensure_columns_exist(columns, frame_name='data')

        self.data = self.data.dropna(subset=columns)

        return self._sync_dataframes()



    def reemplazar_media_columna(self, columns):

        """Reemplazar los valores faltantes con la media de la columna"""

        columns = self._ensure_columns_exist(columns, frame_name='data')

        for column in columns:

            mean_value = self.data[column].mean()

            self.data[column] = self.data[column].fillna(mean_value)

        return self._sync_dataframes()



    def reemplazar_cero_columna(self, columns):

        """Reemplazar los valores faltantes con ceros en las columnas"""

        columns = self._ensure_columns_exist(columns, frame_name='data')

        for column in columns:

            self.data[column] = self.data[column].fillna(0)

        return self._sync_dataframes()



    def reemplazar_texto_columna(self, columns, text):

        """Reemplazar los valores faltantes con un texto especÃ­fico en las columnas"""

        columns = self._ensure_columns_exist(columns, frame_name='data')

        for column in columns:

            self.data[column] = self.data[column].fillna(text)

        return self._sync_dataframes()



    def reemplazar_anterior_columna(self, columns):

        """Reemplazar los valores faltantes con el valor anterior en las columnas"""

        columns = self._ensure_columns_exist(columns, frame_name='data')

        for column in columns:

            self.data[column] = self.data[column].ffill()

        return self._sync_dataframes()



    def reemplazar_siguiente_columna(self, columns):

        """Reemplazar los valores faltantes con el valor siguiente en las columnas"""

        columns = self._ensure_columns_exist(columns, frame_name='data')

        for column in columns:

            self.data[column] = self.data[column].bfill()

        return self._sync_dataframes()



    def interpolar_columna(self, columns):

        """Interpolar los valores faltantes en las columnas"""

        columns = self._ensure_columns_exist(columns, frame_name='data')

        for column in columns:

            self.data[column] = self.data[column].interpolate()

        return self._sync_dataframes()



    def eliminar_columnas_columna(self, columns):

        """Eliminar columnas especÃ­ficas que contienen valores faltantes"""

        columns = self._ensure_columns_exist(columns, frame_name='data')

        self.data = self.data.drop(columns=columns, axis=1)

        return self._sync_dataframes()



    def eliminar_filas_valores_faltantes(self, columns=None):

        """

        Eliminar filas con valores faltantes.

        Si se indican columnas, solo considera esas columnas.

        """

        if columns is not None:

            return self.eliminar_filas_valores_faltantes_en_columnas(columns)

        self.data = self.data.dropna()

        return self._sync_dataframes()



    def eliminar_columnas_valores_faltantes(self):

        """Eliminar todas las columnas que contienen valores faltantes"""

        self.data = self.data.dropna(axis=1)

        return self._sync_dataframes()



    def reemplazar_vecinos(self, columns):

        """Reemplazar los valores faltantes con la media del valor anterior y siguiente"""

        columns = self._ensure_columns_exist(columns, frame_name='data')

        for column in columns:

            self.data[column] = self.data[column].fillna((self.data[column].shift() + self.data[column].shift(-1)) / 2)

        return self._sync_dataframes()



    

    def reemplazar_mediana_columna(self, columns):

        """Reemplazar los valores faltantes con la mediana de las columnas"""

        columns = self._ensure_columns_exist(columns, frame_name='data')

        for column in columns:

            median_value = self.data[column].median()

            self.data[column] = self.data[column].fillna(median_value)

        return self._sync_dataframes()



    def reemplazar_moda_columna(self, columns):

        """Reemplazar los valores faltantes con la moda de las columnas"""

        columns = self._ensure_columns_exist(columns, frame_name='data')

        for column in columns:

            mode_value = self.data[column].mode()[0]

            self.data[column] = self.data[column].fillna(mode_value)

        return self._sync_dataframes()



    def reemplazar_valor_columna(self, columns, value):

        """Reemplazar los valores faltantes con un valor especÃ­fico"""

        columns = self._ensure_columns_exist(columns, frame_name='data')

        for column in columns:

            self.data[column] = self.data[column].fillna(value)

        return self._sync_dataframes()



    def reemplazar_polinomica_columna(self, columns, order=2):

        """Reemplazar los valores faltantes con interpolaciÃ³n polinÃ³mica"""

        columns = self._ensure_columns_exist(columns, frame_name='data')

        for column in columns:

            self.data[column] = self.data[column].interpolate(method='polynomial', order=order)

        return self._sync_dataframes()



    def reemplazar_predictivo_columna(self, columns, target_column):

        """Reemplazar los valores faltantes usando un modelo predictivo"""

        self._ensure_column_exists(target_column, frame_name='data')

        columns = self._ensure_columns_exist(columns, frame_name='data')

        is_null = self.data[target_column].isnull()

        if is_null.any():

            non_null_data = self.data[~is_null]

            null_data = self.data[is_null]

            model = LinearRegression()

            model.fit(non_null_data[columns], non_null_data[target_column])

            self.data.loc[is_null, target_column] = model.predict(null_data[columns])

        return self._sync_dataframes()

    

    def grafico_interactivo(self, columna, outliers):

        """

        Crea un grÃ¡fico de dispersiÃ³n interactivo de la columna proporcionada para visualizar los outliers.

        """

        self._ensure_column_exists(columna, frame_name='df')

        if not outliers:

            logger.info("No outliers to plot for %s", columna)

            return None



        df_outliers = self.df[self.df[columna].isin(outliers)]

        fig = px.scatter(self.df, x=self.df.index, y=columna, title=f"Scatter plot with Outliers in {columna}")

        fig.add_scatter(x=df_outliers.index, y=df_outliers[columna], mode='markers', marker=dict(color='red', size=10), name='Outliers')

        fig.show()

        return fig

    

    #Optimizador



    def reduce_mem_usage(self, df=None, use_float16=False):

        target_df = self.df if df is None else df

        if target_df is None:

            raise ValueError("No DataFrame provided to reduce memory usage.")



        start_mem = target_df.memory_usage(deep=True).sum() / 1024**2

        logger.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



        report = []



        def registrar(columna, dtype_original, dtype_final, status, detalle):

            report.append({

                'column': columna,

                'dtype_original': str(dtype_original),

                'dtype_final': str(dtype_final),

                'status': status,

                'detail': detalle

            })



        def conversion_segura(serie_original, dtype_destino):

            try:

                serie_convertida = serie_original.astype(dtype_destino)

            except (TypeError, ValueError, OverflowError):

                return None, 'conversion_failed'



            memoria_original = serie_original.memory_usage(deep=True)

            memoria_convertida = serie_convertida.memory_usage(deep=True)

            if memoria_convertida >= memoria_original:

                return None, 'no_memory_gain'



            try:

                serie_restaurada = serie_convertida.astype(serie_original.dtype)

            except (TypeError, ValueError, OverflowError):

                return None, 'restore_failed'



            if not serie_original.equals(serie_restaurada):

                return None, 'data_changed'



            return serie_convertida, 'converted'



        def candidatos_enteros(serie):

            tiene_nulos = serie.isnull().any()

            minimo = serie.min(skipna=True)

            if pd.isna(minimo):

                return []



            if tiene_nulos:

                if minimo >= 0:

                    return ['UInt8', 'UInt16', 'UInt32', 'UInt64', 'Int8', 'Int16', 'Int32', 'Int64']

                return ['Int8', 'Int16', 'Int32', 'Int64']



            if minimo >= 0:

                return [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]

            return [np.int8, np.int16, np.int32, np.int64]



        for col in target_df.columns:

            serie = target_df[col]

            col_type = serie.dtype

            convertido = False



            try:

                if pd.api.types.is_bool_dtype(col_type):

                    registrar(col, col_type, col_type, 'skipped', 'bool_dtype')

                    continue



                if pd.api.types.is_integer_dtype(col_type):

                    for dtype in candidatos_enteros(serie):

                        try:

                            if hasattr(col_type, 'itemsize') and np.dtype(dtype).itemsize >= col_type.itemsize:

                                continue

                        except TypeError:

                            pass



                        serie_convertida, motivo = conversion_segura(serie, dtype)

                        if serie_convertida is not None:

                            target_df[col] = serie_convertida

                            registrar(col, col_type, serie_convertida.dtype, 'optimized', 'integer_downcast')

                            convertido = True

                            break



                        if motivo == 'data_changed':

                            break



                    if not convertido:

                        registrar(col, col_type, col_type, 'skipped', 'no_safe_integer_downcast')



                elif pd.api.types.is_float_dtype(col_type):

                    candidatos_float = [np.float32]

                    if use_float16:

                        candidatos_float = [np.float16, np.float32]



                    for dtype in candidatos_float:

                        if np.dtype(dtype).itemsize >= col_type.itemsize:

                            continue

                        serie_convertida, motivo = conversion_segura(serie, dtype)

                        if serie_convertida is not None:

                            target_df[col] = serie_convertida

                            registrar(col, col_type, serie_convertida.dtype, 'optimized', f'float_downcast_to_{np.dtype(dtype).name}')

                            convertido = True

                            break



                        if motivo == 'data_changed':

                            break



                    if not convertido:

                        registrar(col, col_type, col_type, 'skipped', 'no_safe_float_downcast')



                elif pd.api.types.is_object_dtype(col_type):

                    serie_convertida, motivo = conversion_segura(serie, 'category')

                    if serie_convertida is not None:

                        target_df[col] = serie_convertida

                        registrar(col, col_type, serie_convertida.dtype, 'optimized', 'object_to_category')

                    else:

                        registrar(col, col_type, col_type, 'skipped', f'object_not_converted_{motivo}')



                elif isinstance(col_type, pd.CategoricalDtype) or pd.api.types.is_datetime64_any_dtype(col_type):

                    registrar(col, col_type, col_type, 'skipped', 'already_optimized_or_specialized')



                else:

                    registrar(col, col_type, col_type, 'skipped', 'unsupported_dtype')



            except Exception as e:

                registrar(col, col_type, col_type, 'error', str(e))

                logger.error(f'Error occurred when processing column {col}: {e}')



        end_mem = target_df.memory_usage(deep=True).sum() / 1024**2

        logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

        if start_mem > 0:

            logger.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



        self.last_reduce_mem_usage_report = report

        self.df = target_df

        self.data = target_df

        return target_df



    #Extractor



    def extraer_primera_palabra(self, columna, nueva_columna):

        """

        Extrae la primera palabra de una columna separada por comas y la guarda en una nueva columna.

        """

        serie = self._serie_como_texto(columna).apply(lambda value: self._obtener_parte(value, 0))

        return self._sync_text_result(nueva_columna, serie)

    

    def extraer_ultima_palabra(self, columna, nueva_columna):

        """

        Extrae la Ãºltima palabra de una columna separada por comas y la guarda en una nueva columna.

        """

        serie = self._serie_como_texto(columna).apply(lambda value: self._obtener_parte(value, -1))

        return self._sync_text_result(nueva_columna, serie)

    

    def extraer_palabra(self, columna, nueva_columna, numero):

        """

        Extrae la n palabra de una columna separada por comas y la guarda en una nueva columna.

        """

        if numero < 1:

            raise ValueError("numero must be greater than or equal to 1")

        serie = self._serie_como_texto(columna).apply(lambda value: self._obtener_parte(value, numero - 1))

        return self._sync_text_result(nueva_columna, serie)

    

    def extraer_palabra_con_caracter(self, columna, nueva_columna, caracter):

        """

        Extrae la primera palabra que contiene un carÃ¡cter especÃ­fico de una columna separada por comas y la guarda en una nueva columna.

        """

        def extraer(value):

            for palabra in self._split_safe(value):

                if caracter in palabra:

                    return palabra

            return None



        serie = self._serie_como_texto(columna).apply(extraer)

        return self._sync_text_result(nueva_columna, serie)



    def extraer_primera_palabra_mayusculas(self, columna, nueva_columna):

        """

        Convierte la cadena a mayÃºsculas y luego extrae la primera palabra, guardÃ¡ndola en una nueva columna.

        """

        serie = self._serie_como_texto(columna).apply(

            lambda value: None if value is None else self._obtener_parte(value.upper(), 0)

        )

        return self._sync_text_result(nueva_columna, serie)

    

    def reemplazar_nan_y_extraer_primera_palabra(self, columna, nueva_columna):

        """

        Reemplaza valores NaN por una cadena vacÃ­a y luego extrae la primera palabra de una columna separada por comas, guardÃ¡ndola en una nueva columna.

        """

        self._ensure_column_exists(columna, frame_name='df')

        self.df[columna] = self.df[columna].fillna('')

        self.data = self.df

        serie = self.df[columna].apply(lambda value: self._obtener_parte(value, 0))

        return self._sync_text_result(nueva_columna, serie)

    

    def extraer_palabra_especifica(self, columna, nueva_columna, palabra_clave):

        """

        Extrae una palabra especÃ­fica que coincide con un patrÃ³n regex de una columna y la guarda en una nueva columna.

        """

        patron = re.compile(rf'\b{re.escape(str(palabra_clave))}\b')

        serie = self._serie_como_texto(columna).apply(

            lambda value: None if value is None else (patron.search(value).group(0) if patron.search(value) else None)

        )

        return self._sync_text_result(nueva_columna, serie)

    

    def extraer_si_nan_en_otra_columna(self, columna_base, columna_condicion, nueva_columna, unique=False):

        """

        Extrae la primera palabra de una columna si otra columna contiene NaN.

        Si unique=True, obtiene los valores Ãºnicos.

        """

        self._ensure_column_exists(columna_base, frame_name='df')

        self._ensure_column_exists(columna_condicion, frame_name='df')



        filtrado = pd.isna(self.df[columna_condicion])

        serie_base = self._serie_como_texto(columna_base).apply(lambda value: self._obtener_parte(value, 0))



        if unique:

            valores = pd.Series(serie_base[filtrado].dropna().unique(), dtype='object')

            return self._sync_text_result(nueva_columna, valores)

        else:

            self.df.loc[filtrado, nueva_columna] = serie_base[filtrado]

            self.data = self.df

            return self._sync_dataframes()

    

    def extraer_valores_entre_parentesis(self, columna, nueva_columna):

        """

        Extrae valores que estÃ¡n entre parÃ©ntesis en una columna y los guarda en una nueva columna.

        """

        serie = self._serie_como_texto(columna).apply(

            lambda value: None if value is None else (re.search(r'\((.*?)\)', value).group(1) if re.search(r'\((.*?)\)', value) else None)

        )

        return self._sync_text_result(nueva_columna, serie)



    def extraer_numeros(self, columna, nueva_columna):

        """

        Extrae nÃºmeros de una columna y los guarda en una nueva columna.

        """

        serie = self._serie_como_texto(columna).apply(

            lambda value: None if value is None else (re.search(r'(\d+)', value).group(1) if re.search(r'(\d+)', value) else None)

        )

        return self._sync_text_result(nueva_columna, serie)



    def extraer_fechas(self, columna, nueva_columna):

        """

        Extrae fechas de una columna y las guarda en una nueva columna.

        """

        serie = pd.to_datetime(self._serie_como_texto(columna), errors='coerce')

        self.df[nueva_columna] = serie

        self.data = self.df

        return self._sync_dataframes()



    def extraer_correos_electronicos(self, columna, nueva_columna):

        """

        Extrae correos electrÃ³nicos de una columna y los guarda en una nueva columna.

        """

        serie = self._serie_como_texto(columna).apply(

            lambda value: [] if value is None else re.findall(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', value)

        )

        return self._sync_text_result(nueva_columna, serie)



    def extraer_url(self, columna, nueva_columna):

        """

        Extrae URLs de una columna y las guarda en una nueva columna.

        """

        serie = self._serie_como_texto(columna).apply(

            lambda value: [] if value is None else re.findall(r'(https?://\S+)', value)

        )

        return self._sync_text_result(nueva_columna, serie)

    

    # MÃ©todos para detecciÃ³n de outliers

    

    def grubbs_test(self, columna, show_plot=True):

        """

        Realiza el test de Grubbs para detectar outliers (Phoenix).

        """

        x = self._numeric_series(columna).to_numpy()

        n = len(x)

        mean_x = np.mean(x)

        sd_x = np.std(x)



        if n < 3 or sd_x == 0:

            logger.info("No se puede realizar el test de Grubbs en %s", columna)

            return self._registrar_outliers('grubbs_test', columna, [], show_plot=False)



        numerator = max(abs(x - mean_x))

        g_calculated = numerator / sd_x

        t_value = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)

        g_critical = ((n - 1) * np.sqrt(np.square(t_value))) / (np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value)))



        print(f"Grubbs Calculated Value: {g_calculated}")

        print(f"Grubbs Critical Value: {g_critical}")



        outliers = [] if g_critical > g_calculated else [x[np.argmax(abs(x - mean_x))]]

        print("Outliers:", outliers)

        return self._registrar_outliers('grubbs_test', columna, outliers, show_plot=show_plot)



    def zscore_outliers(self, columna, threshold=3, show_plot=True):

        """

        Detecta outliers utilizando el mÃ©todo del Z-score (Phoenix).

        """

        x = self._numeric_series(columna).to_numpy()

        m = np.mean(x)

        sd = np.std(x)



        if sd == 0:

            logger.info("No se puede realizar el test Z-score en %s", columna)

            return self._registrar_outliers('zscore_outliers', columna, [], show_plot=False)



        outliers = [value for value in x if np.abs((value - m) / sd) > threshold]

        print("Outliers:", outliers)

        return self._registrar_outliers('zscore_outliers', columna, outliers, show_plot=show_plot)

    

    def zrscore_outliers(self, columna, threshold=3, show_plot=True):

        """

        Detecta outliers utilizando el mÃ©todo del Z-score robusto (Phoenix).

        """

        def mad(data, axis=None):

            return np.median(np.absolute(data - np.median(data, axis)), axis)



        x = self._numeric_series(columna).to_numpy()

        med = np.median(x)

        mad_val = mad(x)



        if mad_val == 0:

            logger.info("No se puede realizar el test robusto en %s", columna)

            return self._registrar_outliers('zrscore_outliers', columna, [], show_plot=False)



        outliers = [value for value in x if np.abs((0.6745 * (value - med)) / mad_val) > threshold]

        print("Outliers:", outliers)

        return self._registrar_outliers('zrscore_outliers', columna, outliers, show_plot=show_plot)



    def iqr_outliers(self, columna, factor=1.5, show_plot=True):

        """

        Detecta outliers utilizando el mÃ©todo del IQR (Phoenix).

        """

        x = self._numeric_series(columna).to_numpy()

        q1 = np.percentile(x, 25)

        q3 = np.percentile(x, 75)

        iqr = q3 - q1

        lower_bound = q1 - factor * iqr

        upper_bound = q3 + factor * iqr

        outliers = [value for value in x if value < lower_bound or value > upper_bound]

        print("Outliers:", outliers)

        return self._registrar_outliers('iqr_outliers', columna, outliers, show_plot=show_plot)



    def winsorization_outliers(self, columna, lower_percentile=1, upper_percentile=99, show_plot=True):

        """

        Detecta outliers utilizando el mÃ©todo de Winsorization (Phoenix).

        """

        x = self._numeric_series(columna).to_numpy()

        q1 = np.percentile(x, lower_percentile)

        q3 = np.percentile(x, upper_percentile)

        outliers = [value for value in x if value < q1 or value > q3]

        print("Outliers:", outliers)

        return self._registrar_outliers('winsorization_outliers', columna, outliers, show_plot=show_plot)



    def dbscan_outliers(self, columna):

        """

        Detecta outliers utilizando el mÃ©todo DBSCAN (Phoenix).

        """

        x = self._numeric_series(columna).to_numpy().reshape(-1, 1)

        outlier_detection = DBSCAN(eps=2, metric='euclidean', min_samples=5)

        clusters = outlier_detection.fit_predict(x)

        data = pd.DataFrame()

        data['cluster'] = clusters

        self.last_outlier_report = {

            'method': 'dbscan_outliers',

            'column': columna,

            'count': int((clusters == -1).sum()),

            'outliers': x[clusters == -1].flatten().tolist()

        }

        print(data['cluster'].value_counts().sort_values(ascending=False))

        return data



    def isolation_forest_outliers(self, columna):

        """

        Detecta outliers utilizando el mÃ©todo de Isolation Forest (Phoenix).

        """

        x = self._numeric_series(columna).to_numpy().reshape(-1, 1)

        iso = IsolationForest(random_state=1, contamination='auto')

        preds = iso.fit_predict(x)

        data = pd.DataFrame()

        data['cluster'] = preds

        self.last_outlier_report = {

            'method': 'isolation_forest_outliers',

            'column': columna,

            'count': int((preds == -1).sum()),

            'outliers': x[preds == -1].flatten().tolist()

        }

        print(data['cluster'].value_counts().sort_values(ascending=False))

        return data

    

    #Clasificador de variables



    def clasificar_variables(self, cat_threshold=0.05, ordinal_max_unique=15):

        """

        Clasifica las variables en categÃ³ricas, ordinales y continuas de forma robusta.

        """

        variables_categoricas = []

        variables_ordinales = []

        variables_continuas = []

        detalle = {}



        for columna in self.df.columns:

            serie = self.df[columna]

            total = len(serie)

            no_nulos = serie.dropna()

            unicos = self._safe_nunique(no_nulos, dropna=True)

            ratio_unicos = (unicos / total) if total else 0

            dtype = serie.dtype



            if pd.api.types.is_bool_dtype(dtype):

                variables_categoricas.append(columna)

                detalle[columna] = 'categorica_booleana'

            elif isinstance(dtype, pd.CategoricalDtype):

                variables_categoricas.append(columna)

                detalle[columna] = 'categorica_dtype'

            elif pd.api.types.is_object_dtype(dtype):

                if ratio_unicos <= cat_threshold or (unicos <= ordinal_max_unique and unicos < total):

                    variables_categoricas.append(columna)

                    detalle[columna] = 'categorica_texto'

                else:

                    variables_continuas.append(columna)

                    detalle[columna] = 'continua_texto_alta_cardinalidad'

            elif pd.api.types.is_datetime64_any_dtype(dtype):

                variables_continuas.append(columna)

                detalle[columna] = 'continua_temporal'

            elif pd.api.types.is_integer_dtype(dtype):

                if unicos <= ordinal_max_unique and unicos < total:

                    variables_ordinales.append(columna)

                    detalle[columna] = 'ordinal_entera'

                elif ratio_unicos <= cat_threshold:

                    variables_categoricas.append(columna)

                    detalle[columna] = 'categorica_entera'

                else:

                    variables_continuas.append(columna)

                    detalle[columna] = 'continua_entera'

            elif pd.api.types.is_float_dtype(dtype):

                if ratio_unicos <= cat_threshold and unicos <= ordinal_max_unique:

                    variables_categoricas.append(columna)

                    detalle[columna] = 'categorica_numerica'

                else:

                    variables_continuas.append(columna)

                    detalle[columna] = 'continua_flotante'

            else:

                variables_continuas.append(columna)

                detalle[columna] = 'continua_no_clasificada'



        self.last_variable_classification_report = detalle

        logger.info("Variables categoricas: %s", variables_categoricas)

        logger.info("Variables ordinales: %s", variables_ordinales)

        logger.info("Variables continuas: %s", variables_continuas)

        logger.info("Detalle de clasificacion: %s", detalle)

        return variables_categoricas, variables_ordinales, variables_continuas


__all__ = ["Phoenix"]

