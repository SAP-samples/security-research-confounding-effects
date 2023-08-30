#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Overxfl0w13 - 2015 - https://github.com/overxfl0w - #
# For anything -> https://github.com/overxfl0w/CObfuscator #

import re
import sys
from random import randint

""" Se compilarán las expresiones antes del finditer """

non_start_text = "(?!(?:\/\/|\"|\'|\/\*\*))"  # Arreglar #
comments = "(\/\/.*\n|\/\*\s*\n*\s*(?:.*)\s*\n*\s*\*\/)"  # Arreglar #
identifier = "[a-zA-Z][\w|_]*"
declarator = "(?:\*?)\s*" + "(" + identifier + ")"
storage_class_specifiers = "auto|register|static|extern|typedef"
type_qualifiers = "const|volatile"
type_specifiers = "void|char|short|int|long|float|double|signed|unsigned|struct|union"  # Anyadir soporte enum (identifier) #
storage_class_specifiers_functions = "auto|static|extern"
type_qualifiers_functions = type_qualifiers
type_specifiers_functions = "char|short|int|long|float|double|signed|unsigned"
declaration_specifiers = "(?:" + storage_class_specifiers + ")?\s*(?:" + type_qualifiers + ")?\s*(?:" + type_specifiers + ")\*?\s+"
function_definition = non_start_text + declaration_specifiers + declarator + "\s*\("
start_preprocessor = "(?#)"
non_start_preprocessor = "(?!#)"
preprocessor_defines = "define|ifdef|ifndef|undef"
preprocessor = non_start_text + start_preprocessor + "\s*(?:" + preprocessor_defines + ")\s*" + declarator + "\s*\(?"
variable_definition = non_start_text + declaration_specifiers + declarator + "\s*(?:[=|;|,])"
separated_variables = non_start_text + identifier + "\s*,\s*" + declarator + "\s*(?:[=|;|,])"  # Arreglar para variables declaradas con , #
token_numbers = non_start_text + "((?:0x|0b|0x%)?[0-9]+)"
MAX_VALID_IDENTIFIER = 10
JUNK_MAX_FUNCTIONS = 15
JUNK_MAX_PARAMS_PER_FUNCTION = 5
JUNK_MAX_INSTRUCTIONS = 10


def addJunk(source, functions):
    """ Añade código basura con instrucciones que alteren el contenido binario sin alterar el resultado final.
    Devuelve el código fuente con basura añadida
    Parámetros:
    source -- código fuente
    Excepciones:
    ...
    """

    ## Arreglar ##
    def addJunkInstructions(source, functions):
        """ Añade código basura con instrucciones que alteren el contenido binario sin alterar el resultado final.
            Devuelve el código fuente con basura añadida
            Parámetros:
            source -- código fuente
            Excepciones:
            ...
        """
        function_block = None
        for function in functions: function_block = non_start_text + "(?:.*)" + function + "(?:.*)\){"
        return source

    def addJunkFunctions(source):
        """ Añade código basura con instrucciones que alteren el contenido binario sin alterar el resultado final.
            Devuelve el código fuente con basura añadida
            Parámetros:
            source -- código fuente
            Excepciones:
            ...
        """
        numFunctions = randint(1, JUNK_MAX_FUNCTIONS)
        astorage_specifiers = storage_class_specifiers_functions.split("|")
        astorage_specifiers.append("")
        atype_qualifiers = type_qualifiers_functions.split("|")
        atype_qualifiers.append("")
        atype_specifiers = type_specifiers_functions.split("|")
        for i in range(numFunctions):
            my_storage_specifier = astorage_specifiers[randint(0, len(astorage_specifiers) - 1)]
            my_type_qualifier = atype_qualifiers[randint(0, len(atype_qualifiers) - 1)]
            my_type_specifier = atype_specifiers[randint(0, len(atype_specifiers) - 1)]
            len_name = randint(3, MAX_VALID_IDENTIFIER)
            my_name = generateValidIdentifier(len_name)
            num_params = randint(1, JUNK_MAX_PARAMS_PER_FUNCTION)
            source_function = my_storage_specifier + " " + my_type_qualifier + " " + my_type_specifier + " " + my_name + "("
            source_function += atype_specifiers[randint(0, len(atype_specifiers) - 1)] + " " + generateValidIdentifier(
                randint(0, MAX_VALID_IDENTIFIER))
            for x in range(num_params - 1):  source_function += "," + atype_specifiers[
                randint(0, len(atype_specifiers) - 1)] + " " + generateValidIdentifier(randint(0, MAX_VALID_IDENTIFIER))
            source_function += "){}" + "\n" * randint(1, 6)
            source += source_function
        return source

    source = addJunkFunctions(source)
    source = addJunkInstructions(source, functions)
    return source


""" Number codifications functions (Documentar e implementar) """


def convertDecimal(number):
    """ Obtiene el código fuente del fichero.
    Devuelve el código fuente con las declaraciones cambiadas.
    Parámetros:
    source -- código fuente
    Excepciones:
    ...
    """
    return number


def convertBinary(number):
    """ Obtiene el código fuente del fichero.
    Devuelve el código fuente con las declaraciones cambiadas.
    Parámetros:
    source -- código fuente
    Excepciones:
    ...
    """
    try:
        return bin(int(number, 0))
    except:
        return number


def convertHexadecimal(number):
    """ Obtiene el código fuente del fichero.
    Devuelve el código fuente con las declaraciones cambiadas.
    Parámetros:
    source -- código fuente
    Excepciones:
    ...
    """
    try:
        return hex(int(number, 0))
    except:
        return number


def convertOctal(number):
    """ Obtiene el código fuente del fichero.
    Devuelve el código fuente con las declaraciones cambiadas.
    Parámetros:
    source -- código fuente
    Excepciones:
    ...
    """
    try:
        return oct(int(number, 0))
    except:
        return number


def neutralOpNumber(number):
    """ Obtiene el código fuente del fichero.
    Devuelve el código fuente con las declaraciones cambiadas.
    Parámetros:
    source -- código fuente
    Excepciones:
    ...
    """
    # Generar todas las combinaciones #
    neutralOps = [" | 0b0", " | 0b00", " | 0b000", " | 0b0000", " | 0x0", " | 0x00", " | 0", " & 0xF", " & 0xFF",
                  " & 0b11111111", " ^ " + str(number),
                  " + 0b0", " + 0", " + 0b00", " - 0", " - 0b00", " * 0b1", " * 0b01", " / 0b01", " / 0b1"]
    return number + neutralOps[randint(0, len(neutralOps) - 1)]


""" File functions """


def getSource(fileName):
    """ Obtiene el código fuente del fichero.
    Devuelve el código fuente con las declaraciones cambiadas.
    Parámetros:
    source -- código fuente
    Excepciones:
    ...
    """
    with open(fileName, "r") as fd: yield fd.read()
    fd.close()


def saveSource(source, fileName):
    """ Almacena en disco el código fuente modificado.
    No devuelve ningún valor
    Parámetros:
    source   -- código fuente
    fileName -- nombre del fichero donde almacenar el código fuente
    Excepciones:
    ...
    """
    with open(fileName, "w") as fd: fd.write(source)
    fd.close()


""" Replace functions """


def replaceFunctions(source, functions):
    """ Reemplazar funciones haciéndolas ilegibles.
    Devuelve el codigo fuente con las funciones cambiadas
    Parámetros:
    source    -- código fuente
    functions -- conjunto de funciones del código fuente
    Excepciones:
    ...
    Sentencia:
    storage_class_identifier type_qualifier type_specifier
    """
    for function in functions: source = re.sub(r'\b' + function + r'\b', functions[function], source)
    return source


def replaceDeclarations(source, declarations):
    """ Reemplazar variables|funciones definidas con llamadas al preprocesador.
    Devuelve el código fuente con las declaraciones cambiadas
    Parámetros:
    source       -- código fuente
    declarations -- conjunto de declaraciones a reemplazar
    Excepciones:
    ...
    """
    for declaration in declarations: source = re.sub(r'\b' + declaration + r'\b', declarations[declaration], source)
    return source


def replaceVariables(source, variables):
    """ Cambia el nombre de todas las variables haciéndolas ilegibles.
    Devuelve el código fuente con las variables cambiadas
    Parámetros:
    source    -- código fuente
    variables -- conjunto de variables a reemplazar
    Excepciones:
    ...
    """
    for variable in variables: source = re.sub(r'\b' + variable + r'\b', variables[variable], source)
    return source


def replaceNumberCodification(source, numbers):
    """ Obtiene el código fuente del fichero.
    Devuelve el código fuente con las declaraciones cambiadas.
    Parámetros:
    source -- código fuente
    Excepciones:
    ...
    """
    for number in numbers: source = re.sub(r'\b' + number + r'\b', numbers[number], source)
    return source


""" Remove functions """


def removeLR(source):
    """ Obtiene el código fuente del fichero.
    Devuelve el código fuente con las declaraciones cambiadas.
    Parámetros:
    source -- código fuente
    Excepciones:
    ...
    """
    source = source.split("\n")
    for x in range(len(source)):
        if source[x].strip().startswith("#") == True: source[x] += "\n"
    return "".join(source)


def removeComments(source):
    """ Obtiene el código fuente del fichero.
    Devuelve el código fuente con las declaraciones cambiadas.
    Parámetros:
    source -- código fuente
    Excepciones:
    ...
    """
    source = re.sub(comments, "", source)
    return source


""" Search functions """


def searchFunctions(source, lengthValidIdentifiers):
    """ Buscar las funciones y almacenarlas en una tabla hash.
    Devuelve una tabla hash con todas las funciones del código como clave y su token aleatorio como valor
    Parámetros:
    source                 -- código fuente
    lengthValidIdentifiers -- longitud de los identificadores aleatorios
    Excepciones:
    1) La expresion regular no encuentra ocurrencias, match.group(0) = None (AttributeError)
    2) Se sobrepasa el indice del grupo (IndexError)
    ...
    """
    try:
        functions = {}  # Nombre función : aleatorio #
        # print non_start_text+function_definition
        reCompiled = re.compile(function_definition)
        for match in reCompiled.finditer(source):
            funcToken = match.group(1)
            if funcToken != "main":
                if funcToken not in functions: functions[funcToken] = generateValidIdentifier(lengthValidIdentifiers)
    except AttributeError as ae:
        print("AttributeError")
    except IndexError as ie:
        print("Index error")
    finally:
        return functions


def searchDeclarations(source, lengthValidIdentifiers):
    """ Buscar las declaraciones
    Devuelve una tabla hash con todas las declaraciones del código como clave y su token aleatorio como valor.
    Parámetros:
    source                 -- código fuente
    lengthValidIdentifiers -- longitud de los identificadores aleatorios
    Excepciones:
    1) La expresion regular no encuentra ocurrencias, match.group(0) = None (AttributeError)
    2) Se sobrepasa el indice del grupo (IndexError)
    ...
    """
    try:
        declarations = {}  # Nombre declaracion : aleatorio #
        # print preprocessor
        reCompiled = re.compile(preprocessor)
        for match in reCompiled.finditer(source):
            decToken = match.group(1)
            if decToken not in declarations: declarations[decToken] = generateValidIdentifier(lengthValidIdentifiers)
    except AttributeError as ae:
        print("AttributeError")
    except IndexError as ie:
        print("Index error")
    finally:
        return declarations


def searchVariables(source, lengthValidIdentifiers):
    """ Buscar las variables y almacenarlas en una tabla hash.
    Devuelve una tabla hash con todas las variablesdel código como clave y su token aleatorio como valor
    Parámetros:
    source                 -- código fuente
    lengthValidIdentifiers -- longitud de los identificadores aleatorios
    Excepciones:
    1) La expresion regular no encuentra ocurrencias, match.group(0) = None (AttributeError)
    2) Se sobrepasa el indice del grupo (IndexError)
    ...
    """
    try:
        variables = {}  # Nombre declaracion : aleatorio #
        # print variable_definition,separated_variables
        reCompiled = re.compile(variable_definition)
        for match in reCompiled.finditer(source):
            varToken = match.group(1)
            if varToken not in variables: variables[varToken] = generateValidIdentifier(lengthValidIdentifiers)
        ## Arreglar ##
        """for match in re.finditer(separated_variables,source):
            varToken = match.group(1)
            if varToken not in variables: variables[varToken] = generateValidIdentifier(lengthValidIdentifiers)"""
    except AttributeError as ae:
        print("AttributeError")
    except IndexError as ie:
        print("Index error")
    finally:
        return variables


def searchNumbers(source):
    """ Obtiene el código fuente del fichero.
    Devuelve el código fuente con las declaraciones cambiadas.
    Parámetros:
    source -- código fuente
    Excepciones:
    ...
    """
    try:
        numbers = {}  # Nombre función : aleatorio #
        convertFunctions = [convertDecimal, convertBinary, convertHexadecimal, convertOctal, neutralOpNumber]
        reCompiled = re.compile(token_numbers)
        for match in reCompiled.finditer(source):
            numToken = match.group(1)
            if numToken not in numbers:
                randomFunction = randint(0, len(convertFunctions) - 1)
                numbers[numToken] = convertFunctions[randomFunction](numToken)
    except AttributeError as ae:
        print("AttributeError")
    except IndexError as ie:
        print("Index error")
    finally:
        return numbers


def permuteSource(source):
    """ Permuta grupos de instrucciones sin dependencias de flujo ni antidependencias.
    Devuelve el código fuente con grupos de instrucciones con el orden cambiado aleatoriamente
    Parámetros:
    source -- código fuente
    Excepciones:
    ...
    """

    def permuteFunctions(source):
        """ Permuta grupos de instrucciones sin dependencias de flujo ni antidependencias.
        Devuelve el código fuente con grupos de instrucciones con el orden cambiado aleatoriamente
        Parámetros:
        source -- código fuente
        Excepciones:
        ...
        """
        pass

    def permuteInstructions(source):
        """ Permuta grupos de instrucciones sin dependencias de flujo ni antidependencias.
        Devuelve el código fuente con grupos de instrucciones con el orden cambiado aleatoriamente
        Parámetros:
        source -- código fuente
        Excepciones:
        ...
        """
        pass

    pass


def addJumps(source):
    """ Cambia el flujo de ejecución añadiendo saltos sin alterar el resultado final.
    Devuelve el código fuente con saltos añadidos aleatoriamente
    Parámetros:
    source -- código fuente
    Excepciones:
    ...
    """
    pass


def transposeSource(source):
    """ Transponer el código manteniendo el flujo de ejecución.
    Devuelve el código fuente transpuesto aleatoriamente
    Parámetros:
    source -- código fuente
    Excepciones:
    ...
    """


def replaceInstructions(source):
    """ Reemplaza instrucciones por otras equivalentes.
    Devuelve el código fuente con las instrucciones reemplazadas aleatoriamente
    Parámetros:
    source -- código fuente
    Excepciones:
    ...
    """
    pass


""" Utils functions """


def generateValidIdentifier(length):
    """ Generar un identificador valido para las sustituciones.
    Devuelve un identificador aleatorio x / |x| = length, iniciado por un caracter alfabético seguido de length-1 caracteres alfanuméricos
    Parámetros:
    length -- longitud del identificador
    Excepciones:
    ...
    """
    return "".join([chr(randint(65, 90)) if randint(0, 2) >= 1 else chr(randint(97, 122))] + [
        chr(randint(65, 90)) if randint(0, 2) == 2 else chr(randint(97, 122)) if randint(0, 2) == 1 else chr(
            randint(48, 57)) for x in range(length)])


if __name__ == "__main__":
    ## source = getSource("mutation_motor.c").next() ##

    """ Código de ejemplo hardcodeado """

    source = open(sys.argv[1]).read()

    numbers = searchNumbers(source)
    source = replaceNumberCodification(source, numbers)
    functions = searchFunctions(source, 10)
    source = replaceFunctions(source,functions)
    declarations = searchDeclarations(source,10)
    source = replaceDeclarations(source,declarations)
    variables = searchVariables(source,10)
    source = replaceVariables(source,variables)
    source = addJunk(source, functions)
    source = removeComments(source)
    source = removeLR(source)

    print(source)
