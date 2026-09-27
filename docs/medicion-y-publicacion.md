**Medición y publicación del blog**

La integración de Google Analytics queda preparada y desactivada: `services.googleAnalytics.id` sigue vacío. La compilación actual no carga Google Analytics ni el script de eventos opcionales.

**Cuando tengas una propiedad GA4**

Configura el flujo web de tu propiedad y utiliza su ID de medición `G-…`. Puedes completar `id` bajo `[services.googleAnalytics]` en `hugo.toml`, o proporcionar la variable de entorno `HUGO_SERVICES_GOOGLEANALYTICS_ID` en tu proceso de construcción. El ID identifica la propiedad; no es una clave secreta. No publiques un identificador de ejemplo.

Construye de nuevo en modo producción y verifica en GA4 que llega una visita y que funcionan los eventos esperados. Revisa la configuración de privacidad y consentimiento que corresponda a tu sitio antes de activar un servicio de medición.

| Evento preparado | Cuándo se emite | Qué no demuestra |
|---|---|---|
| `consultation_contact` | Clic en un enlace de email desde Work with me | Que se envió un correo o existe una venta |
| `email_contact` | Clic en email desde otra página | Que se recibió un mensaje |
| `study_download` | Clic en un ZIP del mismo sitio | Que finalizó la descarga o se usó el material |
| `newsletter_submit` | Envío válido del formulario de Buttondown | Que se confirmó la suscripción |

Los eventos personalizados no leen ni envían el email escrito por el visitante; incluyen la ruta de la página y, cuando corresponde, la ruta del recurso o ubicación del formulario. Respetan Do Not Track y Global Privacy Control. Son adicionales al comportamiento del proveedor GA4, que debe configurarse por separado.

Comprueba altas reales en Buttondown. No envíes promociones de cursos a una lista cuya promesa sigue siendo recibir ensayos sin promociones. Las consultas profesionales utilizan la página Work with me.

**Search Console**

Agrega y verifica el dominio en tu propia cuenta. La verificación DNS se configura en el proveedor del dominio; el repo no puede completarla por sí solo. Si eliges una propiedad con prefijo de URL y metaetiqueta, PaperMod admite `params.analytics.google.SiteVerificationTag`. Usa el valor real que proporciona Google.

Envía el sitemap de producción: `https://carlosdanieljimenez.com/sitemap.xml`. El archivo `robots.txt` se genera con Hugo. Verifica la indexación de la portada, Start Here y los artículos centrales después de publicar. La existencia del sitemap no garantiza indexación ni posiciones en buscadores.

**Construcción y publicación**

El sitio utiliza Hugo y sirve los archivos compilados desde la raíz del repositorio. Las fuentes están en `content/`, `layouts/`, `assets/css/extended/` y `static/`. `public/` y los HTML de la raíz son resultados de construcción.

Construcción local:

```sh
hugo --cleanDestinationDir --minify
```

Vista previa:

```sh
hugo server --environment production
```

Si no quieres cargar un ID de Analytics ya configurado durante una vista previa, utiliza el entorno de desarrollo predeterminado (`hugo server`). El entorno de producción es útil para comprobar metadatos sociales.

El script existente `deploy.sh` construye, copia `public/*` a la raíz, ejecuta `git add -A`, crea un commit y hace push a `master`. Revisa el diff y las inclusiones antes de usarlo: también incluiría cambios ajenos que estén presentes en el directorio de trabajo. Para publicar los cambios ya revisados:

```sh
./deploy.sh "Clarify positioning, research evidence, and consulting offer"
```

Una compilación local o copiar los archivos a la raíz no publica en GitHub Pages. La publicación requiere el push y la ejecución correcta del despliegue de GitHub.

**Materiales editables**

- `scripts/create_social_cards.py`: genera cinco tarjetas PNG con Pillow y fuentes Georgia/Arial o DejaVu. No se necesita ejecutarlo en cada compilación: los PNG están en `static/img/social/`.
- `scripts/audit_editorial_exports.py`: inspecciona archivos agregados existentes y genera un resumen de procedencia. No realiza llamadas a modelos.
- `docs/visibilidad-linkedin-x.md`: perfiles, calendario y borradores para ambas redes, con enlaces UTM.

Se comprobó la construcción con Hugo 0.166.0. PaperMod emite avisos de deprecación de propiedades de idioma con esta versión; no impiden la construcción. La revisión editorial no equivale a una auditoría científica completa ni una reproducción de modelos.
