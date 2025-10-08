# The Probability and the Word

Blog personal sobre MLOps, LLMs y APIs construido con Hugo y desplegado en GitHub Pages.

🌐 **URL**: https://carlosdanieljimenez.com/

## 📋 Requisitos Previos

- [Hugo](https://gohugo.io/installation/) instalado en tu sistema
- Git configurado con acceso a GitHub
- Cuenta de GitHub con permisos de escritura en el repositorio

## 🏗️ Estructura del Proyecto

Este es un blog Hugo con la siguiente estructura:

```
.
├── content/          # Contenido del blog (posts, páginas)
│   ├── about/       # Página "Acerca de"
│   ├── archive/     # Archivo de posts
│   ├── notes/       # Notas
│   └── post/        # Posts del blog
├── static/          # Archivos estáticos (imágenes, etc.)
├── themes/          # Tema Hugo (hugo-theme-cleanwhite)
├── public/          # Sitio generado (NO MODIFICAR DIRECTAMENTE)
├── hugo.toml        # Configuración de Hugo
└── deploy.sh        # Script de despliegue automático
```

## ✍️ Flujo de Trabajo para Publicar Cambios

### Opción 1: Despliegue Automático (Recomendado)

Usa el script `deploy.sh` que automatiza todo el proceso:

```bash
# Hacer cambios en los archivos de contenido (content/, static/, etc.)

# Ejecutar el script de despliegue con un mensaje de commit
./deploy.sh "Mensaje descriptivo de tus cambios"

# O sin mensaje (usará timestamp automático)
./deploy.sh
```

El script automáticamente:
1. Construye el sitio con Hugo
2. Hace commit de los cambios en el directorio `public/`
3. Hace push a GitHub Pages

### Opción 2: Despliegue Manual Paso a Paso

Si prefieres control total sobre el proceso:

#### 1. Editar Contenido

```bash
# Crear un nuevo post
hugo new post/nombre-del-post.md

# O editar archivos existentes en content/
vim content/post/mi-post.md
vim content/about/index.md
```

#### 2. Previsualizar Localmente (Opcional pero Recomendado)

```bash
# Iniciar servidor de desarrollo
hugo server -D

# Abrir en el navegador: http://localhost:1313
```

#### 3. Generar el Sitio Estático

```bash
# Construir el sitio (genera archivos en public/)
hugo
```

#### 4. Desplegar a GitHub Pages

**IMPORTANTE**: El directorio `public/` es un repositorio Git separado que apunta al mismo repositorio pero contiene el sitio compilado.

```bash
# Navegar al directorio public
cd public/

# Agregar todos los cambios
git add .

# Hacer commit con un mensaje descriptivo
git commit -m "Actualización del blog: descripción de cambios"

# Subir a GitHub
git push origin master

# Volver al directorio raíz
cd ..
```

#### 5. Guardar Cambios del Código Fuente

```bash
# En el directorio raíz del proyecto
git add .

# Commit de los archivos fuente
git commit -m "Actualización de contenido: descripción"

# Push al repositorio principal
git push origin master
```

## 🔄 Verificar que los Cambios se Reflejen

1. **Espera 2-5 minutos**: GitHub Pages tarda un poco en procesar y desplegar los cambios

2. **Verifica el despliegue en GitHub**:
   - Ve a: https://github.com/carlosjimenez88M/carlosjimenez88m.github.io
   - Pestaña "Actions" (si está habilitado)
   - O verifica la fecha del último commit en la rama `master`

3. **Limpia la caché del navegador**:
   ```bash
   # Chrome/Brave: Cmd+Shift+R (Mac) o Ctrl+Shift+R (Windows/Linux)
   # Safari: Cmd+Option+E y luego Cmd+R
   ```

4. **Visita el sitio**: https://carlosdanieljimenez.com/

## 🚨 Solución de Problemas Comunes

### Los cambios no se reflejan en el sitio web

**Causa**: No has construido y desplegado el directorio `public/`

**Solución**:
```bash
# Reconstruir el sitio
hugo

# Desplegar public/
cd public/
git add .
git commit -m "rebuild site"
git push origin master
cd ..
```

### Error "nothing to commit, working tree clean"

**Causa**: No has hecho cambios en los archivos fuente antes de hacer commit

**Solución**: Primero edita archivos en `content/`, `static/`, etc., luego construye con `hugo` y despliega

### Hugo no está instalado o no se encuentra

**Solución en macOS**:
```bash
brew install hugo
```

**Solución en Linux**:
```bash
# Ubuntu/Debian
sudo apt install hugo

# O descarga desde: https://gohugo.io/installation/
```

### El sitio se ve roto después de actualizar

**Causa**: Archivos estáticos o configuración incorrecta

**Solución**:
```bash
# Limpiar archivos generados
rm -rf public/ resources/

# Reconstruir desde cero
hugo

# Desplegar
cd public/
git add .
git commit -m "rebuild site"
git push origin master
```

## 📝 Comandos Útiles de Hugo

```bash
# Crear nuevo post
hugo new post/mi-nuevo-post.md

# Servidor de desarrollo con drafts
hugo server -D

# Construir para producción
hugo

# Construir incluyendo drafts
hugo -D

# Ver versión de Hugo
hugo version

# Limpiar cache
hugo mod clean
```

## 🔧 Configuración del Blog

La configuración principal está en `hugo.toml`:
- URL base: `https://carlosdanieljimenez.com/`
- Tema: `hugo-theme-cleanwhite`
- Búsqueda con Algolia (configurar credenciales si se usa)

## 📞 Contacto

- **Email**: danieljimenez88m@gmail.com
- **Twitter**: [@DanielJimenezM9](https://x.com/DanielJimenezM9)
- **GitHub**: [@carlosjimenez88M](https://github.com/carlosjimenez88M)

## 📄 Licencia

Este blog es un proyecto personal. El contenido es propiedad de Carlos Daniel Jiménez.
