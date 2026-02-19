# --- PASO 1: Primeras líneas de código ---
# Imprimir un mensaje en la consola
print("Hello, World")

# --- PASO 2 Y 3: Creación de objetos y asignación ---
# Asignación del nombre y edad usando los datos proporcionados
my_name <- "Jose"
my_age <- 22
asd <-1
# El operador "flecha" también puede usarse hacia la derecha
"Ming" -> my_name2

# Mostrar el contenido de los objetos en la consola
print(my_name)
my_name2
my_age

# Listar todos los objetos guardados actualmente en la memoria
ls()

# --- PASO 4: Tipos de datos (Building Blocks) ---
# Carácter (Texto)
class(my_name)

# Numérico (Números)
my_birth_year <- 2003
class(my_birth_year)

# Combinar varios elementos en un vector con la función c()
nombres <- c("Jose", "Ming")
ciudades <- c("Barcelona", "Madrid", "Valencia")

# Factor (Variables categóricas)
education_group <- factor(c("low", "medium", "high", "low"))
class(education_group)

# Valores lógicos (TRUE/FALSE)
is_student <- c(TRUE, FALSE, TRUE)
class(is_student)

# --- PASO 5: Matemáticas, Lógica y Funciones ---
# Comparaciones lógicas (devuelven TRUE o FALSE)
my_age == 22  # ¿Es exactamente igual a 22?
my_age != 30  # ¿Es diferente de 30?
my_age > 18   # ¿Es mayor que 18?

# R como calculadora
2026 - 2003   # Cálculo simple

# Crear secuencias de números
# Genera números del 1 al 20, de uno en uno
mis_numeros <- seq(from = 1, to = 20, by = 1)
mis_numeros

# Estadísticas básicas de resumen
sum(mis_numeros)     # Suma total
mean(mis_numeros)    # Media aritmética
median(mis_numeros)  # Mediana
sd(mis_numeros)      # Desviación estándar
max(mis_numeros)     # Valor máximo
min(mis_numeros)     # Valor mínimo

# Uso de la función rep() para repetir valores
# Repetir el nombre "Jose" tres veces
rep("Jose", times = 3)

# Funciones anidadas (combinar varias funciones en una línea)
# Crea secuencia del 1 al 10, calcula su media y la repite 3 veces
rep(mean(seq(from = 1, to = 10, by = 1)), times = 3)

# Versión paso a paso de lo anterior para mayor claridad
paso_1 <- seq(from = 1, to = 10, by = 1)
paso_2 <- mean(paso_1)
paso_3 <- rep(paso_2, times = 3)
paso_3

# --- LIMPIEZA DEL ESPACIO DE TRABAJO ---
# Eliminar todos los objetos cargados en el Environment
# Use esto con precaución
remove(list = ls())