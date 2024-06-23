import copy

vocales_no_acentuadas = ['a', 'e', 'i', 'o', 'u']
vocales_acentuadas = ['á', 'é', 'í', 'ó', 'ú']
de_no_a_si_acentuadas = {'a': 'á', 'e': 'é', 'i': 'í', 'o': 'ó', 'u': 'ú', 'ï': 'ï', 'ü': 'ü'}
de_si_a_no_acentuadas = {valor: clave for clave, valor in de_no_a_si_acentuadas.items()}
dieresis = ['ï', 'ü']
vocales = vocales_no_acentuadas + vocales_acentuadas + dieresis
vocales_y = vocales + ['y']
diptongos = ['ai', 'au', 'ei', 'eu', 'oi', 'ou', 'ui', 'iu', 'ia', 'ua', 'ie', 'ue', 'io', 'uo', 'ió', 'ey', 'oy', 'ié',
             'éi', 'ué', 'ái', 'iá', 'uá']
diptongo_artificial = {'ai': 'ái', 'au': 'áu', 'ei': 'éi', 'eu': 'éu', 'oi': 'ói', 'ou': 'óu', 'ui': 'uí', 'iu': 'iú',
                       'ia': 'iá', 'ua': 'uá', 'ie': 'ié', 'ue': 'ué', 'io': 'ió', 'uo': 'uó'}

FACTOR_AGUDA, FACTOR_LLANA, FACTOR_ESDRUJULA = 1, 0, -1


def remove_dots(text, sep=' '):
    quitar = [':', '\'', ',', '.', ';', '.', '–', '(', ')', '\n', '\r', '¿', '?', '!', '¡', '—', '»', '”', '“', '«', '-', '/',
              '/']
    for q in quitar:
        text = text.replace(q, sep)
    return text

def remove_nums(text):
    return ''.join(caracter for caracter in text if not caracter.isdigit())

def normalizar(text):
    """Quita puntuación, mayusculas y espacios laterales

        Args:
            texto (str): La palabra a normalizar
        Returns:
            str: palabra sin puntuación, mayusculas y espacios laterales
    """
    return remove_dots(remove_nums(text)).strip().lower()


def normalizar_qu_gu(palabra):
    """Normaliza palabras con digrafos qu y gu, para que no se tenga en cuenta la vocal del digrafo en el cómputo

        Args:
            palabra (str): La palabra con posible digrafo
        Returns:
            str: palabra sin el digrafo
    """
    quitar = [('qu', 'q'), ('gue', 'ge'), ('gui', 'gi')]
    for q in quitar:
        palabra = palabra.replace(q[0], q[1])
    return palabra

def get_stress_position(palabra):
    """Normaliza palabras con digrafos qu y gu, para que no se tenga en cuenta la vocal del digrafo en el cómputo

        Args:
            palabra (str): La palabra de la que extraer el acento
        Returns:
            int: posición del acento
    """
    pos_acento = 0
    # si una sola letra
    if len(palabra) == 1:
        pos_acento = 0
    elif not palabra:
        pos_acento = 0
    else:
        num_silabas = 0
        palabra = normalizar_qu_gu(palabra)
        acento = None
        pos_vocales = []
        for i, c in enumerate(palabra):
            if c in vocales:
                num_silabas += 1
                if palabra[i - 1] + c in diptongos and not palabra[i - 1] in dieresis and num_silabas > 1:
                    num_silabas -= 1
                else:
                    pos_vocales.append(i)
                # tilde
                if c in vocales_acentuadas:
                    pos_acento = i
                    acento = num_silabas

        # como todas las esdrújulas se acentúan...
        if acento is None:
            if len(pos_vocales) == 0:
                pos_acento = 0
            elif len(pos_vocales) == 1:
                pos_acento = pos_vocales[0]
            else:
                if palabra[-1] in ['n', 's'] + vocales:
                    acento = num_silabas - 1
                    pos_acento = pos_vocales[-2]
                # aguda
                else:
                    acento = num_silabas
                    pos_acento = pos_vocales[-1]
    return pos_acento

def get_rhymes(word):
    
    pos_acento = get_stress_position(word)

    # nos quedamos con la palabra desde la última vocal acentuada para la rima consonante   
    consonante = word[pos_acento:]
    consonante_aso = copy.copy(consonante)

    # OPERACIONES SOBRE LA RIMA CONSONANTE
    # si la primera posicion del acento es una vocal no tildada, la tildamos para la salida del sistema
    if consonante[0] in vocales_no_acentuadas:
        consonante = de_no_a_si_acentuadas[consonante[0]]+consonante[1:]

    # las rimas consonantes deben coincidir con la sílaba acentuada en los diptongos, por eso hay que hacer este tratamiento adicional
    for k, v in diptongo_artificial.items():
        if consonante_aso.find(k) > -1:
            consonante_aso = consonante_aso.replace(k,v)
            consonante = consonante_aso[consonante_aso.index(v)+1:]
    
    # OPERACIONES SOBRE LA RIMA ASONANTE
    asonancia = ""
    primera = True    
    for i, c in enumerate(consonante_aso):
        if c in vocales:
            # diptongo cerrada+abierta, como vamos a comprobar las vocales contiguas, debemos comprobar primero que no nos salgamos del array
            if i + 1 < len(consonante_aso) and c in vocales_no_acentuadas and c + consonante_aso[i + 1] in diptongos:
                continue
            # diptongo abierta+cerrada
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
            asonancia = asonancia[0]+de_si_a_no_acentuadas[asonancia[-1]]

    return {'palabra':word, 
            'pos_acento': pos_acento, 
            'consonante': consonante, 
            'asonante': asonancia}

def get_rhymes_vowels(word):
    # si una sola letra
    if len(word) == 1:
        return word
    rima = ""
    for c in word[::-1]:
        if c in vocales:
            rima += c
        if len(rima) >= 2:
            return rima[::-1]
    return rima[::-1]

def get_rhymes_lines(lines):
    rimas = []
    for line in lines:
        line = normalizar(line)
        if not line:
            continue        
        print(line)
        rimas.append(get_rhymes(line.split()[-1]))
    return rimas

def get_rhyme_scheme(poem):
    # obtemos las rimas asonantes y consonantes que se da en el poema
    todas_rimas = get_rhymes_lines(poem)
    # extraemos el esquema de un tipo de rima
    rimas = [rima['asonante'] for rima in todas_rimas]
    esquema = ['' for _ in rimas]
    char = 'A'
    # para cada rima única de la lista
    for i_rima in list(dict.fromkeys(rimas)):
        # buscamos cada rima que coincida
        for j, j_rima in enumerate(rimas):
            if i_rima == j_rima:
                esquema[j] = char
        # pasamos a la siguiente rima única        
        char = chr(ord(char) + 1) 
    
    rima_con = {}
    for i, i_letra in enumerate(esquema):
        palabra_a_rimar = todas_rimas[i]['palabra']
        rima_con[palabra_a_rimar] = {}
        for j, j_letra in enumerate(esquema):
            if i_letra == j_letra:
                palabra_con_que_rima = todas_rimas[j]['palabra']
                rima_con[palabra_a_rimar][j] = palabra_con_que_rima

    esquema_txt = ''.join(esquema)

    return {'esquema':esquema_txt, 'rimas_vocales':rimas, 'rima_con':rima_con}



