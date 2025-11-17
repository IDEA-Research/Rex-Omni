import os
from flask import Flask, render_template, request, redirect, flash
from rex_omni import RexOmniWrapper, RexOmniVisualize
from PIL import Image


UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'EZEc6*AASw&&Zx'

#Initialize the Rex-Omni OCR model
model_path = "IDEA-Research/Rex-Omni"

rex_model = RexOmniWrapper(
    model_path=model_path,
    backend="vllm",  # Choose "transformers" or "vllm"
    max_tokens=2048,
    temperature=0.0,
    top_p=0.05,
    top_k=1,
    repetition_penalty=1.05,
)

def get_text_from_image(filepath):
    try:
        image = Image.open(filepath).convert("RGB")
        categories = ["word"]

        results = rex_model.inference(images=image, task="ocr_box", categories=categories)
        result = results[0]

        raw_output = results[0]['raw_output']

        if result["success"]:
            predictions = result["extracted_predictions"]
            vis_image_pil = RexOmniVisualize(
                image=image,
                predictions=predictions,
                font_size=20,
                draw_width=5,
                show_labels=True,
            )
            
            return vis_image_pil, raw_output
        else:
            return None, raw_output

    except Exception as e:
        print(f"Eroare in timpul procesarii OCR: {e}")
        return None, f"Eroare la procesarea imaginii: {e}"


@app.route("/", methods=['GET', 'POST'])
def index():
    original_filename = None
    visualized_filename = None
    raw_output = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Niciun fișier selectat')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('Niciun fișier selectat')
            return redirect(request.url)
        
        if file:
            original_filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            
            file.save(filepath)
            
            flash('Imaginea a fost încărcată cu succes!')

            try:
                vis_image_pil, raw_output_text = get_text_from_image(filepath)
                raw_output = raw_output_text

                if vis_image_pil:
                    name, ext = os.path.splitext(original_filename)
                    visualized_filename = f"{name}_ocr{ext}"
                    visualized_filepath = os.path.join(app.config['UPLOAD_FOLDER'], visualized_filename)
                    
                    vis_image_pil.save(visualized_filepath)
                    flash('Imaginea a fost incarcata si procesata cu succes!')
                else:
                    flash('Imaginea a fost incarcata, dar procesarea a esuat.')

            except Exception as e:
                flash(f'A aparut o eroare in timpul procesarii: {e}')
                raw_output = f"Eroare: {e}"
            
            return render_template('index.html', 
                                   original_filename=original_filename,
                                   visualized_filename=visualized_filename,
                                   raw_output=raw_output)

        
    return render_template('index.html', 
                           original_filename=None, 
                           visualized_filename=None, 
                           raw_output=None)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", debug=True)