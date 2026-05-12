import glob # Búsqueda de carpetas de checkpoints por patrón de nombre
import shutil # Eliminación recursiva de directorios

# FUNCIONES
# Elimina todas las carpetas de checkpoints de NEAT Snake del directorio actual
def limpiar_checkpoints():
    carpetas_val = glob.glob('checkpoints-neat-snake*')

    if not carpetas_val:
        print("No se encontraron checkpoints. Nada que limpiar.")

        return

    for carpeta_iter in carpetas_val:
        shutil.rmtree(carpeta_iter)
        print(f"Eliminada: {carpeta_iter}")

    print("\nListo. El próximo entrenamiento comenzará desde Generación 0.")

# PUNTO DE PARTIDA
confirmacion = input("¿Borrar todo el progreso de evolución? (s/n): ")

if confirmacion.strip().lower() == 's':
    limpiar_checkpoints()
else:
    print("Cancelado.")

input()
