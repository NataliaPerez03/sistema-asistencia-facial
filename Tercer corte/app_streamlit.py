import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
from PIL import Image
import io

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Asistencia",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #9333EA;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">📸 Sistema de Asistencia - Reconocimiento Facial</h1>', unsafe_allow_html=True)

# Variables globales
PATH_IMAGES = "ImagesAttendance"
PATH_CSV = "Attendance.csv"

# Crear carpeta si no existe
if not os.path.exists(PATH_IMAGES):
    os.makedirs(PATH_IMAGES)


# Función para cargar encodings
@st.cache_resource
def cargar_encodings():
    """Carga los encodings de todas las imágenes."""
    encode_list = []
    class_names = []
    estudiantes_unicos = {}

    if not os.path.exists(PATH_IMAGES):
        return encode_list, class_names

    myList = os.listdir(PATH_IMAGES)

    for cl in myList:
        if not cl.endswith(('.jpg', '.jpeg', '.png')):
            continue

        curImg = cv2.imread(f'{PATH_IMAGES}/{cl}')
        if curImg is None:
            continue

        img_rgb = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img_rgb)

        if len(face_locations) == 0:
            continue

        try:
            encode = face_recognition.face_encodings(img_rgb, face_locations)[0]
            nombre_archivo = os.path.splitext(cl)[0]

            if '_' in nombre_archivo and nombre_archivo.split('_')[-1].isdigit():
                nombre_base = '_'.join(nombre_archivo.split('_')[:-1])
            else:
                nombre_base = nombre_archivo

            encode_list.append(encode)
            class_names.append(nombre_base)

            if nombre_base not in estudiantes_unicos:
                estudiantes_unicos[nombre_base] = 0
            estudiantes_unicos[nombre_base] += 1

        except Exception as e:
            st.error(f"Error al procesar {cl}: {str(e)}")

    return encode_list, class_names


# Cargar encodings
encode_list_known, class_names = cargar_encodings()

# Sidebar - Menú
st.sidebar.title("📋 Menú Principal")
menu = st.sidebar.radio(
    "Selecciona una opción:",
    ["🏠 Inicio", "📸 Registrar Estudiante", "🎥 Procesar Video", "📊 Ver Registros"]
)

# ==================== SECCIÓN: INICIO ====================
if menu == "🏠 Inicio":
    st.subheader("👋 Bienvenido al Sistema de Asistencia")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("👥 Estudiantes Registrados", len(set(class_names)))

    with col2:
        st.metric("📸 Total de Fotos", len(class_names))

    with col3:
        archivos_csv = len([f for f in os.listdir('.') if f.endswith('.csv')])
        st.metric("📄 Registros de Asistencia", archivos_csv)

    st.info(
        "📌 **Instrucciones:**\n\n1. **Registrar Estudiante:** Captura fotos de nuevos estudiantes\n2. **Procesar Video:** Detecta estudiantes en un video de clase\n3. **Ver Registros:** Consulta la asistencia registrada")

# ==================== SECCIÓN: REGISTRAR ESTUDIANTE ====================
elif menu == "📸 Registrar Estudiante":
    st.subheader("📸 Registrar Nuevo Estudiante")

    nombre = st.text_input("Nombre del estudiante:", placeholder="Ej: Juan Pérez")

    st.info(
        "📋 **Instrucciones:**\n\n1. Escribe el nombre del estudiante\n2. Sube varias fotos con diferentes variaciones:\n   - Sin accesorios\n   - Con gafas\n   - Con gorro\n   - Diferentes ángulos")

    uploaded_files = st.file_uploader(
        "Sube las fotos del estudiante:",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )

    if st.button("💾 Guardar Estudiante", type="primary"):
        if not nombre:
            st.error("⚠️ Por favor ingresa el nombre del estudiante")
        elif not uploaded_files:
            st.error("⚠️ Por favor sube al menos una foto")
        else:
            # Guardar fotos
            contador = 1
            archivos_existentes = [f for f in os.listdir(PATH_IMAGES) if f.startswith(nombre) and f.endswith('.jpg')]
            contador = len(archivos_existentes) + 1

            for uploaded_file in uploaded_files:
                img = Image.open(uploaded_file)
                img_array = np.array(img)

                # Verificar que tenga rostro
                face_locations = face_recognition.face_locations(img_array)

                if len(face_locations) == 0:
                    st.warning(f"⚠️ No se detectó rostro en: {uploaded_file.name}")
                    continue

                # Guardar imagen
                img_name = f"{nombre}_{contador}.jpg"
                img_path = os.path.join(PATH_IMAGES, img_name)
                img.save(img_path)
                contador += 1

            # Limpiar caché para recargar
            st.cache_resource.clear()

            st.success(f"✅ Se guardaron {contador - len(archivos_existentes) - 1} foto(s) de {nombre}")
            st.balloons()

# ==================== SECCIÓN: PROCESAR VIDEO ====================
elif menu == "🎥 Procesar Video":
    st.subheader("🎥 Procesar Video de Asistencia")

    if len(class_names) == 0:
        st.warning("⚠️ No hay estudiantes registrados. Por favor registra estudiantes primero.")
    else:
        st.info(f"👥 Estudiantes registrados: **{len(set(class_names))}**")

        uploaded_video = st.file_uploader("Sube el video de la clase:", type=['mp4', 'avi', 'mov', 'mkv'])

        if uploaded_video is not None:
            if st.button("▶️ Procesar Video", type="primary"):
                with st.spinner("🔄 Procesando video..."):
                    # Guardar video temporalmente
                    temp_video_path = "temp_video.mp4"
                    with open(temp_video_path, "wb") as f:
                        f.write(uploaded_video.read())

                    # Procesar video
                    cap = cv2.VideoCapture(temp_video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))

                    estudiantes_detectados = set()
                    frame_count = 0
                    skip_frames = max(fps, 30)

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1

                        # Actualizar progreso
                        if frame_count % 10 == 0:
                            progreso = min(frame_count / total_frames, 1.0)
                            progress_bar.progress(progreso)
                            status_text.text(f"Frame {frame_count} / {total_frames}")

                        # Saltar frames
                        if frame_count % skip_frames != 0:
                            continue

                        # Detectar rostros
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_frame)

                        if len(face_locations) == 0:
                            continue

                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                        for encoding in face_encodings:
                            matches = face_recognition.compare_faces(encode_list_known, encoding, tolerance=0.5)

                            if True in matches:
                                match_index = matches.index(True)
                                nombre = class_names[match_index]
                                estudiantes_detectados.add(nombre)

                    cap.release()
                    os.remove(temp_video_path)

                    # Mostrar resultados
                    st.success(f"✅ Procesamiento completado")
                    st.metric("👥 Estudiantes detectados", f"{len(estudiantes_detectados)} de {len(set(class_names))}")

                    if len(estudiantes_detectados) > 0:
                        st.subheader("📋 Lista de Asistencia:")
                        for i, nombre in enumerate(sorted(estudiantes_detectados), 1):
                            st.success(f"✓ {i}. {nombre}")

                        # Guardar CSV
                        if st.button("💾 Guardar en CSV"):
                            fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            nombre_archivo = f"Asistencia_Video_{fecha_hora}.csv"

                            with open(nombre_archivo, 'w', encoding='utf-8') as f:
                                f.write("Nombre,Fecha,Hora,Presente\n")
                                fecha = datetime.now().strftime("%Y-%m-%d")
                                hora = datetime.now().strftime("%H:%M:%S")

                                for nombre in sorted(estudiantes_detectados):
                                    f.write(f"{nombre},{fecha},{hora},Sí\n")

                            st.success(f"✅ Guardado como: {nombre_archivo}")
                    else:
                        st.warning("❌ No se detectaron estudiantes en el video")

# ==================== SECCIÓN: VER REGISTROS ====================
elif menu == "📊 Ver Registros":
    st.subheader("📊 Registros de Asistencia")

    archivos_csv = [f for f in os.listdir('.') if f.startswith('Asistencia') and f.endswith('.csv')]

    if len(archivos_csv) == 0:
        st.info("No hay registros de asistencia aún")
    else:
        archivo_seleccionado = st.selectbox("Selecciona un registro:", archivos_csv)

        if archivo_seleccionado:
            import pandas as pd

            df = pd.read_csv(archivo_seleccionado)
            st.dataframe(df, use_container_width=True)

            st.download_button(
                label="📥 Descargar CSV",
                data=open(archivo_seleccionado, 'rb').read(),
                file_name=archivo_seleccionado,
                mime="text/csv"
            )
