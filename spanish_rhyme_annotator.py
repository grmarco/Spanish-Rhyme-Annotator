import json
import copy
import spacy
import itertools
from string import ascii_uppercase
from collections import Counter
import pandas as pd #Importamos pandas

# Constantes de vocales y transformaciones de tildes
vocales_no_acentuadas = ['a', 'e', 'i', 'o', 'u']
vocales_acentuadas = ['á', 'é', 'í', 'ó', 'ú']
de_no_a_si_acentuadas = {'a': 'á', 'e': 'é', 'i': 'í', 'o': 'ó', 'u': 'ú', 'ï': 'ï', 'ü': 'ü'}
de_si_a_no_acentuadas = {valor: clave for clave, valor in de_no_a_si_acentuadas.items()}
dieresis = ['ï', 'ü']
vocales = vocales_no_acentuadas + vocales_acentuadas + dieresis
vocales_y = vocales + ['y']

# Diptongos y sus versiones con tilde
diptongos = [
    'ai', 'au', 'ei', 'eu', 'oi', 'ou', 'ui', 'iu', 'ia', 'ua', 'ie', 'ue', 'io', 'uo',
    'ió', 'ey', 'oy', 'ié', 'éi', 'ué', 'ái', 'iá', 'uá'
]
diptongo_artificial = {
    'ai': 'ái', 'au': 'áu', 'ei': 'éi', 'eu': 'éu', 'oi': 'ói', 'ou': 'óu', 'ui': 'uí', 'iu': 'iú',
    'ia': 'iá', 'ua': 'uá', 'ie': 'ié', 'ue': 'ué', 'io': 'ió', 'uo': 'uó'
}

FACTOR_AGUDA, FACTOR_LLANA, FACTOR_ESDRUJULA = 1, 0, -1

def generate_rhyme_labels():
    """
    Generador que produce etiquetas de rima secuenciales:
    'A', 'B', ..., 'Z', 'AA', 'AB', ..., 'ZZ', 'AAA', etc.
    """
    length = 1
    while True:
        for label in itertools.product(ascii_uppercase, repeat=length):
            yield ''.join(label)
        length += 1

class SpanishRhymeAnnotator:
    def __init__(self):
        pass

    @staticmethod
    def remove_dots(text, sep=' '):
        quitar = [
            ':', '\'', ',', '.', ';', '.', '–', '(', ')', '\n', '\r', '¿', '?', '!', '¡', '—', '»',
            '”', '“', '«', '-', '/', '/','…'
        ]
        for q in quitar:
            text = text.replace(q, sep)
        return text

    @staticmethod
    def remove_nums(text):
        return ''.join(car for car in text if not car.isdigit())

    @staticmethod
    def normalizar(text):
        """
        Quita puntuación, mayúsculas y espacios laterales.
        """
        return SpanishRhymeAnnotator.remove_dots(SpanishRhymeAnnotator.remove_nums(text)).strip().lower()

    @staticmethod
    def tipo_palabra(palabra):
        vocales_acent = "áéíóú"

        def silaba_tonica(p):
            for i, letra in enumerate(reversed(p)):
                if letra in vocales_acent:
                    return len(p) - i
            return -1

        pos_tonica = silaba_tonica(palabra)
        if pos_tonica == -1:
            if palabra[-1] in "nsaeiou":
                pos_tonica = len(palabra) - 1
            else:
                pos_tonica = len(palabra)

        if pos_tonica == len(palabra):
            return "aguda"
        elif pos_tonica == len(palabra) - 1:
            return "llana"
        elif pos_tonica <= len(palabra) - 2:
            return "esdrújula"

    @staticmethod
    def normalizar_qu_gu(palabra):
        """
        Normaliza palabras con dígrafos 'qu' y 'gu' para el cómputo.
        """
        quitar = [('qu', 'q'), ('gue', 'ge'), ('gui', 'gi')]
        for q in quitar:
            palabra = palabra.replace(q[0], q[1])
        return palabra

    @staticmethod
    def get_stress_position(palabra):
        """
        Retorna la posición del acento en la palabra.
        """
        if len(palabra) <= 1:
            return 0
        palabra = SpanishRhymeAnnotator.normalizar_qu_gu(palabra)
        num_silabas = 0
        acento = None
        pos_acento = 0
        pos_vocales = []
        for i, c in enumerate(palabra):
            if c in vocales:
                num_silabas += 1
                if i > 0 and palabra[i - 1] + c in diptongos and palabra[i - 1] not in dieresis and num_silabas > 1:
                    num_silabas -= 1
                else:
                    pos_vocales.append(i)
                if c in vocales_acentuadas:
                    pos_acento = i
                    acento = num_silabas
        if acento is None:
            if len(pos_vocales) == 0:
                pos_acento = 0
            elif len(pos_vocales) == 1:
                pos_acento = pos_vocales[0]
            else:
                if palabra[-1] in ['n', 's'] + vocales:
                    acento = num_silabas - 1
                    pos_acento = pos_vocales[-2]
                else:
                    acento = num_silabas
                    pos_acento = pos_vocales[-1]
        return pos_acento

    nlp = spacy.load("es_core_news_sm")

    @staticmethod
    def normalize_rhyme_word(word):
        """
        Normaliza una palabra para la detección de rima.
        """
        word = SpanishRhymeAnnotator.normalizar(word)
        return word

    @staticmethod
    def get_rhymes(word):
        """
        Obtiene la rima consonante y asonante de una palabra.
        """
        word = SpanishRhymeAnnotator.normalize_rhyme_word(word)
        return SpanishRhymeAnnotator.get_rhyme_data(word)

    @staticmethod
    def get_rhyme_data(word):
        """
        Obtiene las rimas consonante y asonante directamente.
        """
        pos_acento = SpanishRhymeAnnotator.get_stress_position(word)
        consonante = word[pos_acento:]
        consonante_aso = copy.copy(consonante)

        if consonante and consonante[0] in vocales_no_acentuadas:
            consonante = de_no_a_si_acentuadas[consonante[0]] + consonante[1:]

        for k, v in diptongo_artificial.items():
            if consonante_aso.find(k) > -1:
                consonante_aso = consonante_aso.replace(k, v)
                consonante = consonante_aso[consonante_aso.index(v) + 1:]

        asonancia = ""
        primera = True
        for i, c in enumerate(consonante_aso):
            if c in vocales:
                if i + 1 < len(consonante_aso) and c in vocales_no_acentuadas and c + consonante_aso[i + 1] in diptongos:
                    continue
                elif i - 1 >= 0 and c in vocales_no_acentuadas and consonante_aso[i - 1] + c in diptongos:
                    continue
                else:
                    if primera:
                        if c in vocales_no_acentuadas:
                            asonancia += de_no_a_si_acentuadas[c]
                        else:
                            asonancia += c
                        primera = False
                    else:
                        asonancia += c

        if len(asonancia) > 2:
            asonancia = asonancia[0] + asonancia[-1]

        if len(asonancia) == 2:
            if asonancia[0] in vocales_acentuadas and asonancia[1] in vocales_acentuadas:
                asonancia = asonancia[0] + de_si_a_no_acentuadas[asonancia[-1]]

        return {
            'palabra': word,
            'pos_acento': pos_acento,
            'consonante': consonante,
            'asonante': asonancia
        }


    @staticmethod
    def get_rhymes_lines(lines):
        """
        Obtiene las rimas de cada línea, normalizando para singulares/plurales.
        """
        rimas = []
        for line in lines:
            line = SpanishRhymeAnnotator.normalizar(line)
            if not line:
                continue
            palabra_rima = line.split()[-1]
            palabra_rima = SpanishRhymeAnnotator.normalizar_fonemas(palabra_rima)
            rima = SpanishRhymeAnnotator.get_rhymes(palabra_rima)
            rima['palabra_original'] = palabra_rima  # Guardamos la palabra original para la comparación con "s"
            rimas.append(rima)
        return rimas


    @staticmethod
    def normalizar_fonemas(palabra_rima):
        palabra_rima = palabra_rima.replace('v', 'b')
        palabra_rima = palabra_rima.replace('y', 'i')
        return palabra_rima

    @staticmethod
    def get_rhyme_scheme(poem):
        """
        Obtiene el esquema de rima de un poema, asignando etiquetas solo a rimas compartidas por al menos dos versos.
        """
        # Separar versos
        lista_versos = poem.split('\n')
        todas_rimas = SpanishRhymeAnnotator.get_rhymes_lines(lista_versos)

        # Extraer rimas consonantes y asonantes
        rimas_consonantes = [rima['consonante'] for rima in todas_rimas]
        rimas_asonantes = [rima['asonante'] for rima in todas_rimas]
        palabras_originales = [rima['palabra_original'] for rima in todas_rimas]
        
        esquema = ['' for _ in rimas_consonantes]
        current_label = generate_rhyme_labels()

        rima_con = {}
        
        verse_words = [line for line in lista_versos if line.strip()]

        for i, (rima_consonante, rima_asonante, palabra_original, rima_info) in enumerate(zip(rimas_consonantes, rimas_asonantes, palabras_originales, todas_rimas)):
            if esquema[i]:
                continue

            # 1. Buscar rimas consonantes clásicas
            versos_que_riman_consonante = [j for j in range(len(rimas_consonantes)) if rimas_consonantes[j] == rima_consonante]

            # 2. Buscar rimas consonantes con "s" (singular/plural)
            versos_que_riman_con_s = []
            es_rima_con_s = False  # Flag para indicar si se encontró rima con "s"
            if palabra_original[-1] == 's':
                palabra_sin_s = palabra_original[:-1]
                rima_sin_s = SpanishRhymeAnnotator.get_rhymes(palabra_sin_s)['consonante']
                versos_que_riman_con_s = [j for j in range(len(rimas_consonantes)) if rimas_consonantes[j] == rima_sin_s]
                if versos_que_riman_con_s and versos_que_riman_consonante!=versos_que_riman_con_s:
                    es_rima_con_s = True
            

            # 3. Asignar etiqueta si hay rima (con "s" tiene prioridad)
            if len(versos_que_riman_con_s) > 1:
                
                #verificamos que sean palabras del mismo lema. 
                palabras_rima = [todas_rimas[j]['palabra_original'] for j in versos_que_riman_con_s]
                doc = SpanishRhymeAnnotator.nlp(palabras_rima[0])
                if doc:
                     lema_palabra = doc[0].lemma_
                     if all(SpanishRhymeAnnotator.nlp(p)[0].lemma_==lema_palabra for p in palabras_rima):
                            for label, info in rima_con.items():
                                if info['tipo'] == 'consonante' and info['rima'] == rima_consonante and isinstance(info.get('palabras_originales'),list) and any(p.startswith(palabra_original[:-1]) or p.startswith(palabra_original) for p in info['palabras_originales']):
                                        esquema[i] = label
                                        for j in versos_que_riman_con_s:
                                            esquema[j] = label
                                        break
                            else:
                                label = next(current_label)
                                esquema[i] = label
                                rima_con[label] = {
                                    'tipo': 'consonante',
                                    'rima': rima_consonante,
                                    'palabras_originales': [todas_rimas[j]['palabra_original'] for j in versos_que_riman_con_s],
                                    'rima_con_s': es_rima_con_s
                                }
                                for j in versos_que_riman_con_s:
                                    esquema[j] = label

            elif len(versos_que_riman_consonante) > 1:
                for label, info in rima_con.items():
                    if info['tipo'] == 'consonante' and info['rima'] == rima_consonante:
                        esquema[i] = label
                        for j in versos_que_riman_consonante:
                            esquema[j] = label
                        break
                else:
                    label = next(current_label)
                    esquema[i] = label
                    rima_con[label] = {
                        'tipo': 'consonante',
                        'rima': rima_consonante,
                        'palabras_originales': [todas_rimas[j]['palabra_original'] for j in versos_que_riman_consonante],
                        'rima_con_s': es_rima_con_s
                   }
                    for j in versos_que_riman_consonante:
                        esquema[j] = label

            elif len( [j for j in range(len(rimas_asonantes)) if rimas_asonantes[j] == rima_asonante]) > 1:
                for label, info in rima_con.items():
                     if info['tipo'] == 'asonante' and info['rima'] == rima_asonante:
                        esquema[i] = label
                        for j in  [j for j in range(len(rimas_asonantes)) if rimas_asonantes[j] == rima_asonante]:
                            esquema[j] = label
                        break
                else:
                    label = next(current_label)
                    esquema[i] = label
                    rima_con[label] = {
                        'tipo': 'asonante',
                        'rima': rima_asonante,
                        'palabras_originales': [todas_rimas[j]['palabra_original'] for j in [j for j in range(len(rimas_asonantes)) if rimas_asonantes[j] == rima_asonante]],
                         'rima_con_s': es_rima_con_s
                    }
                    for j in [j for j in range(len(rimas_asonantes)) if rimas_asonantes[j] == rima_asonante]:
                         esquema[j] = label
            else:
                 # Si no se encontró rima ni consonante ni asonante, buscamos si existe una rima mixta.
                
                for j in range(len(todas_rimas)):
                  if i !=j: 
                     if todas_rimas[i]['consonante'] == todas_rimas[j]['consonante'] or todas_rimas[i]['asonante'] == todas_rimas[j]['asonante']:
                         for label, info in rima_con.items():
                            if info['tipo'] == 'mixta' and (info['rima_consonante'] == rima_consonante or info['rima_asonante']==rima_asonante):
                                        esquema[i] = label
                                        esquema[j] = label
                                        break
                         else:
                            label = next(current_label)
                            esquema[i] = label
                            rima_con[label] = {
                                'tipo': 'mixta',
                                'rima_consonante': rima_consonante,
                                 'rima_asonante':rima_asonante,
                                'palabras_originales': [todas_rimas[i]['palabra_original'],todas_rimas[j]['palabra_original']],
                                'rima_con_s': es_rima_con_s
                            }
                            esquema[j]=label
                         break

        # Crear la salida
        resultado = {
            'esquema': ''.join(esquema),
            'rima_con': rima_con
        }
        return resultado
    
    @staticmethod
    def detect_rhyme_pattern_general(rhyme, stanza_length=4, patterns=None):
        if patterns is None:
            patterns = {
                'Monorhy': ['AAAA'],
                'Alternating': ['ABAB'],
                'Chained': ['AABB', 'ABBA', 'ABABAB', 'ABCABCABC'],
                'Tail': ['ABCABC']
            }

        counts = {name: 0 for name in patterns}
        counts['Others'] = 0
        num_stanzas = len(rhyme) // stanza_length

        for i in range(num_stanzas):
            stanza = rhyme[i*stanza_length:(i+1)*stanza_length]
            matched = False
            for name, pattern_list in patterns.items():
                for p in pattern_list:
                    if len(p) == len(stanza) and SpanishRhymeAnnotator.match_pattern(stanza, p):
                        counts[name] += 1
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                counts['Others'] += 1
        return counts

    @staticmethod
    def match_pattern(stanza, pattern):
        """
        Verifica si una estrofa coincide con un patrón de rima dado.
        """
        if len(stanza) != len(pattern):
            return False
        
        mapeo = {}  # Diccionario para mapear letras a rimas
        for i, (letra_patron, letra_estrofa) in enumerate(zip(pattern, stanza)):
            if letra_patron not in mapeo:
                mapeo[letra_patron] = letra_estrofa
            elif mapeo[letra_patron] != letra_estrofa:
                return False
        return True
    
    @staticmethod
    def analyze_rhyme_features(poem):
        """
        Analiza varias características de la rima en un poema dado.

        Returns:
        dict: Un diccionario con el análisis de las características de la rima.
        """
        rhyme_data = SpanishRhymeAnnotator.get_rhyme_scheme(poem)
        rhyme_scheme = rhyme_data['esquema']
        rhymes_info = rhyme_data['rima_con']
        
         # Separar versos
        lista_versos = poem.split('\n')

        # 1. Frecuencia de rimas plural-singular
        plural_singular_count = sum(1 for info in rhymes_info.values() if info.get('rima_con_s', False))
        
         # Obtener los versos que riman en plural/singular
        plural_singular_verses = []
        for label, info in rhymes_info.items():
             if info.get('rima_con_s', True):
                if len(info['palabras_originales'])>1:
                     plural_singular_verses.append(info['palabras_originales'])
        
        # 2. Estructuras de rima simétricas
        stanza_length = 4  # Asumimos cuartetas
        symmetric_patterns = SpanishRhymeAnnotator.detect_rhyme_pattern_general(rhyme_scheme, stanza_length, {'Alternating': ['ABAB'], 'Chained': ['AABB', 'ABBA']})
        
        # 3. Rima y ritmo
        ripio_count = SpanishRhymeAnnotator.count_repetition_in_rhyme(rhymes_info)
        internal_rhyme_count = SpanishRhymeAnnotator.count_internal_rhymes(poem)
        average_verse_length = SpanishRhymeAnnotator.average_verse_length(poem)

        # 4. Tipos de rima (consonante, asonante, mixta)
        rhyme_types_analysis = SpanishRhymeAnnotator.analyze_rhyme_types(rhymes_info)

        # 5. Monorrimas extensivas
        monorrimas_extensivas = SpanishRhymeAnnotator.find_monorrimas(rhyme_scheme, lista_versos)

        # 6. Rimas idénticas
        identical_rhymes = SpanishRhymeAnnotator.find_identical_rhymes(rhymes_info)

        analysis = {
            'plural_singular_count': plural_singular_count,
            'plural_singular_verses' : plural_singular_verses,
            'symmetric_rhyme_patterns': symmetric_patterns,
            'ripio_count': ripio_count,
            'internal_rhyme_count': internal_rhyme_count,
            'average_verse_length': average_verse_length,
            'rhyme_types_analysis': rhyme_types_analysis,
             'monorrimas_extensivas': monorrimas_extensivas,
            'identical_rhymes': identical_rhymes
        }
        return analysis
    
    @staticmethod
    def count_repetition_in_rhyme(rhymes_info):
         """
        Cuenta la frecuencia de rimas repetitivas.
        El objetivo principal de esta función es identificar cuántas veces se utilizan las mismas rimas (consonantes o asonantes) más de una vez en un poema. Esto puede indicar un uso excesivo o poco creativo de la rima, lo que en ocasiones se critica como "ripio".
        """
         
         rima_counts = Counter(info['rima'] if info['tipo'] != 'mixta' else (info['rima_consonante'] if info.get('rima_consonante') else info['rima_asonante']) for info in rhymes_info.values())
         repetition_count = sum(count - 1 for count in rima_counts.values() if count > 1)
         return repetition_count

    @staticmethod
    def count_internal_rhymes(poem):
        """
        Cuenta la frecuencia de rimas internas dentro de los versos (tanto consonantes como asonantes).
        """
        lines = poem.split('\n')
        internal_rhyme_count = 0
        for line in lines:
           words = SpanishRhymeAnnotator.normalizar(line).split()
           if len(words) > 2:
            for i in range(len(words)-1):
                for j in range(i+1,len(words)):
                    rima_1 = SpanishRhymeAnnotator.get_rhymes(words[i])
                    rima_2 = SpanishRhymeAnnotator.get_rhymes(words[j])
                    if rima_1['consonante'] == rima_2['consonante'] or rima_1['asonante'] == rima_2['asonante']:
                        internal_rhyme_count+=1
                    
        return internal_rhyme_count


    @staticmethod
    def average_verse_length(poem):
        """
        Calcula la longitud promedio de los versos.
        """
        lines = poem.split('\n')
        total_length = sum(len(line.split()) for line in lines if line)
        num_lines = len([line for line in lines if line])
        average_length = total_length / num_lines if num_lines > 0 else 0
        return average_length
    
    @staticmethod
    def analyze_rhyme_types(rhymes_info):
        """
        Analiza y cuenta los tipos de rima (consonante, asonante, mixta).
        """
        rhyme_types = {'consonante': 0, 'asonante': 0, 'mixta': 0}
        for info in rhymes_info.values():
           if info['tipo'] == 'consonante':
              rhyme_types['consonante'] +=1
           elif info['tipo'] == 'asonante':
               rhyme_types['asonante'] += 1
           elif info['tipo'] == 'mixta':
               rhyme_types['mixta'] +=1
        return rhyme_types
    
    @staticmethod
    def find_monorrimas(rhyme_scheme, lista_versos, stanza_length=4):
        """
        Identifica monorrimas extensivas en un poema.

        Parámetros:
            rhyme_scheme (str): Esquema de rima del poema.
            lista_versos (list): Lista de versos del poema.
            stanza_length (int): Longitud de las estrofas (por defecto, 4).

        Retorna:
            list: Lista de estrofas con monorrimas extensivas.
        """
        monorrimas = []

        if not rhyme_scheme or not lista_versos:
            return monorrimas  # Retorna vacío si no hay esquema de rima o versos

        # Iterar a través de las estrofas
        for i in range(0, len(lista_versos), stanza_length):
            estrofa = lista_versos[i:i + stanza_length]
            esquema_estrofa = rhyme_scheme[i:i + stanza_length]

            # Validar que la estrofa tiene la longitud esperada y que no hay etiquetas vacías
            if len(estrofa) == len(esquema_estrofa) and all(esquema_estrofa):
                # Si todas las rimas de la estrofa son iguales, es una monorrima
                if len(set(esquema_estrofa)) == 1:
                    monorrimas.append(estrofa)

        return monorrimas


    @staticmethod
    def find_identical_rhymes(rhymes_info):
         """
        Identifica ejemplos de rimas idénticas (misma palabra rimando).
        """
         identical_rhymes = []
         
         for label, info in rhymes_info.items():
              if len(info['palabras_originales'])>1:
                  if len(set(info['palabras_originales']))==1:
                       identical_rhymes.append(info['palabras_originales'])
         return identical_rhymes
    
    
def load_corpus(filepath):
    """
    Carga el corpus desde un archivo JSON.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    return corpus

def filter_humor_songs(corpus, humor_songs):
    """
    Filtra las canciones humorísticas del corpus.
    """
    humor_corpus = [song for song in corpus if song['cancion'] in humor_songs]
    non_humor_corpus = [song for song in corpus if song['cancion'] not in humor_songs]
    return humor_corpus, non_humor_corpus

def analyze_corpus(corpus):
    """
    Analiza un corpus completo y calcula las métricas de rima.
    """
    all_analysis = []
    for song in corpus:
      if song.get('letra_text'):
        analysis = SpanishRhymeAnnotator.analyze_rhyme_features(song['letra_text'])
        analysis['poem']=song['letra_text'] #Agregamos el poema para calcular tasas luego
        all_analysis.append(analysis)
    return all_analysis

def calculate_average_rhyme_features(all_analysis):
    """
    Calcula las métricas promedio de rima de un conjunto de análisis,
    incluyendo tanto los valores absolutos como las tasas normalizadas,
    y ahora también ejemplos de rimas plurales, monorrimas e idénticas.
    """
    if not all_analysis:
        return {}

    total_songs = len(all_analysis)
    avg_analysis = {
        'avg_plural_singular_count': 0,
        'avg_plural_singular_rate': 0,
        'avg_symmetric_rhyme_patterns_Alternating': 0,
        'avg_symmetric_rhyme_patterns_Alternating_rate': 0,
        'avg_symmetric_rhyme_patterns_Chained': 0,
        'avg_symmetric_rhyme_patterns_Chained_rate': 0,
        'avg_symmetric_rhyme_patterns_Others': 0,
        'avg_symmetric_rhyme_patterns_Others_rate': 0,
        'avg_ripio_count': 0,
        'avg_ripio_rate': 0,
        'avg_internal_rhyme_count': 0,
        'avg_internal_rhyme_rate': 0,
        'avg_average_verse_length': 0,
        'avg_rhyme_types_consonante': 0,
        'avg_rhyme_types_consonante_rate': 0,
        'avg_rhyme_types_asonante': 0,
        'avg_rhyme_types_asonante_rate': 0,
        'avg_rhyme_types_mixta': 0,
        'avg_rhyme_types_mixta_rate': 0,
        'total_monorrimas_extensivas': 0,
        'avg_monorrimas_extensivas_rate': 0,
        'total_identical_rhymes': 0,
        'avg_identical_rhymes_rate': 0,
        'plural_singular_verses_examples': [],
        'monorrimas_extensivas_examples': [],
        'identical_rhymes_examples': []
    }

    for analysis in all_analysis:
        total_rhymes = sum(analysis['rhyme_types_analysis'].values())
        total_words = sum(len(line.split()) for line in analysis['poem'].split('\n') if line)
        total_stanzas = len(analysis['poem'].split('\n'))//4
        total_verses = len([line for line in analysis['poem'].split('\n') if line])

        avg_analysis['avg_plural_singular_count'] += analysis['plural_singular_count']
        if total_verses>0:
            avg_analysis['avg_plural_singular_rate'] += analysis['plural_singular_count']/total_verses
        
        avg_analysis['avg_symmetric_rhyme_patterns_Alternating'] += analysis['symmetric_rhyme_patterns']['Alternating']
        avg_analysis['avg_symmetric_rhyme_patterns_Chained'] += analysis['symmetric_rhyme_patterns']['Chained']
        avg_analysis['avg_symmetric_rhyme_patterns_Others'] += analysis['symmetric_rhyme_patterns']['Others']
        if total_stanzas>0:
            avg_analysis['avg_symmetric_rhyme_patterns_Alternating_rate'] += analysis['symmetric_rhyme_patterns']['Alternating']/total_stanzas
            avg_analysis['avg_symmetric_rhyme_patterns_Chained_rate'] += analysis['symmetric_rhyme_patterns']['Chained']/total_stanzas
            avg_analysis['avg_symmetric_rhyme_patterns_Others_rate'] += analysis['symmetric_rhyme_patterns']['Others']/total_stanzas
       
        avg_analysis['avg_ripio_count'] += analysis['ripio_count']
        if total_rhymes>0:
            avg_analysis['avg_ripio_rate'] += analysis['ripio_count']/total_rhymes
        
        avg_analysis['avg_internal_rhyme_count'] += analysis['internal_rhyme_count']
        if total_words>0:
            avg_analysis['avg_internal_rhyme_rate'] += analysis['internal_rhyme_count']/total_words
        
        avg_analysis['avg_average_verse_length'] += analysis['average_verse_length']
        
        avg_analysis['avg_rhyme_types_consonante'] += analysis['rhyme_types_analysis']['consonante']
        avg_analysis['avg_rhyme_types_asonante'] += analysis['rhyme_types_analysis']['asonante']
        avg_analysis['avg_rhyme_types_mixta'] += analysis['rhyme_types_analysis']['mixta']

        if total_rhymes>0:
            avg_analysis['avg_rhyme_types_consonante_rate'] += analysis['rhyme_types_analysis']['consonante']/total_rhymes
            avg_analysis['avg_rhyme_types_asonante_rate'] += analysis['rhyme_types_analysis']['asonante']/total_rhymes
            avg_analysis['avg_rhyme_types_mixta_rate'] += analysis['rhyme_types_analysis']['mixta']/total_rhymes
        
        avg_analysis['total_monorrimas_extensivas'] += len(analysis['monorrimas_extensivas'])
        if total_stanzas>0:
            avg_analysis['avg_monorrimas_extensivas_rate'] += len(analysis['monorrimas_extensivas'])/total_stanzas
        
        avg_analysis['total_identical_rhymes'] += len(analysis['identical_rhymes'])
        if total_rhymes>0:
           avg_analysis['avg_identical_rhymes_rate'] += len(analysis['identical_rhymes'])/total_rhymes
        
        # Agregar ejemplos
        avg_analysis['plural_singular_verses_examples'].extend(analysis['plural_singular_verses'])
        avg_analysis['monorrimas_extensivas_examples'].extend(analysis['monorrimas_extensivas'])
        avg_analysis['identical_rhymes_examples'].extend(analysis['identical_rhymes'])
        
    for key in avg_analysis:
        if key not in ['plural_singular_verses_examples', 'monorrimas_extensivas_examples', 'identical_rhymes_examples']:
            avg_analysis[key] /= total_songs

    return avg_analysis

def compare_corpus_data(humor_analysis, non_humor_analysis):
    """
    Compara los datos de dos corpus (humor y no humor) y devuelve un DataFrame de pandas.
    Incluye ejemplos de rimas plurales, monorrimas e idénticas.
    """
    humor_avg_analysis = calculate_average_rhyme_features(humor_analysis)
    non_humor_avg_analysis = calculate_average_rhyme_features(non_humor_analysis)

    # Convertir los diccionarios a DataFrames
    humor_df = pd.DataFrame([humor_avg_analysis])
    non_humor_df = pd.DataFrame([non_humor_avg_analysis])

     # Añadir una columna que identifique cada corpus
    humor_df['corpus'] = 'humor'
    non_humor_df['corpus'] = 'non-humor'

    # Concatenar los DataFrames
    comparison_df = pd.concat([humor_df, non_humor_df], ignore_index=True)
    comparison_df = comparison_df.set_index('corpus')
    print({'humor_results':humor_df, 'non_humor_df_results':non_humor_df})
    return comparison_df

def main():
    # 1. Cargar el corpus
    filepath = r"C:\Users\gmarc\OneDrive - UNED\workspace-current\estilo_sabina\data_processed\sabina_por_cancion.json"
    corpus = load_corpus(filepath)

    # 2. Lista de canciones humorísticas
    humor_songs = [
        'Tratado de impaciencia', 
        'Pasándolo bien', 
        'Ocupen su localidad', 
        'Telespañolito', 
        'Eh, Sabina', 
        'Juana la loca',
        'Incompatibilidad de caracteres', 
        'Oiga, doctor',
        'Cuernos', 
        'Peligro de incendio', 
        'Con un par', 'Ataque de tos', 
        'Pastillas para no soñar',
        'El blues de lo que pasa en mi escalera',
        'No soporto el rap', 
        'Pero qué hermosas eran',
        'Como te digo una co te digo la o',
        'Ya eyaculé',
        'Semos diferentes',
        'A vuelta de correo', 
        'Ay, Calixto', 
        'Idiotas, palizas y calientabraguetas'
    ]

    # 3. Filtrar canciones
    humor_corpus, non_humor_corpus = filter_humor_songs(corpus, humor_songs)

    # 4. Analizar ambos corpus
    humor_analysis = analyze_corpus(humor_corpus)
    non_humor_analysis = analyze_corpus(non_humor_corpus)

    # 5. Comparar los corpus
    corpus_comparison = compare_corpus_data(humor_analysis, non_humor_analysis)

    # 6. Imprimir resultados como tabla
    print("\nAnálisis comparativo de corpus:")
    print(corpus_comparison.to_markdown())

    # 7. Exportar a CSV (opcional)
    corpus_comparison.to_csv("corpus_comparison.csv")
    print("\nAnálisis exportado a 'corpus_comparison.csv'")

    # Ejemplo de uso con una cancion en concreto
    cancion_ejemplo = "Y nos dieron las diez"
    cancion_datos = next((item for item in corpus if item["cancion"] == cancion_ejemplo), None)
    if cancion_datos and cancion_datos.get('letra_text'):
        analysis_ejemplo = SpanishRhymeAnnotator.analyze_rhyme_features(cancion_datos['letra_text'])
        print(f"\nAnálisis de '{cancion_ejemplo}':")
        for key, value in analysis_ejemplo.items():
            print(f"  {key}: {value}")
    
if __name__ == "__main__":
    main()