import numpy as np
import io
import base64
from PIL import Image
from tensorflow.python.framework.op_def_library import apply_op
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
img_names = []

for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
    img_names.append(feature_path.stem)
features = np.array(features)

# Finding Images by query image
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":",".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        # L2 distances to features
        dists = np.linalg.norm(features-query, axis=1)
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id], img_names[id]) for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


# Evaluate Image retrieval system
@app.route('/evaluation', methods=['GET', 'POST'])
def evaluation():
    if request.method == 'POST':
        label = request.form['rdlabel']
        ap = []
        queries = []
        map = 0.0

        for query_file_path in sorted(Path("./evaluation/groundtruth").glob("*_query.txt")):
            # e.g., ../evaluation/groundtruth/xxx_query.txt

            query_image_name = file_lines_to_list(query_file_path)[0].split()[
                0].replace("oxc1_", "")
            query_image_path = Path("./static/img") / \
                (query_image_name + ".jpg")
            print("\nQuery: "+str(query_file_path) +
                  "\n===> Caculation AP for: " + str(query_image_name) + ".jpg")

            img = Image.open(query_image_path)  # PIL image
            query = fe.extract(img)
            # L2 distances to features
            dists = np.linalg.norm(features-query, axis=1)
            ids = np.argsort(dists)[:30]  # Top 30 results
            results = [img_names[id] for id in ids]
            dataLabel = file_lines_to_list(
                str(query_file_path).replace("query", label))

            # ap = average_precision_score(results, goods)
            ap.append(caculate_AP(results, dataLabel))
            queries.append(str(query_image_name))

        map = round(cal_average(ap), 4)

        print("\n===> MAP: " + str(map))

        plt.barh(queries, ap, color="forestgreen")

        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(ap):
            str_val = " " + str(val) # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color='forestgreen', va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(ap)-1): # largest bar
                adjust_axes(r, t, fig, axes)
        # set window title
        fig.canvas.set_window_title("Mean Average Precision")
        # write classes in y axis
        tick_font_size = 10
        # plt.yticks(range(ap))
        """
        Re-scale height accordingly
        """
        init_height = fig.get_figheight()
        # comput the matrix height in points and inches
        dpi = fig.dpi
        height_pt = len(queries) * (tick_font_size * 0.8) # 0.8 (some spacing)
        height_in = height_pt / dpi
        # compute the required figure height 
        top_margin = 0.5 # in percentage of the figure height
        bottom_margin = 0.1 # in percentage of the figure height
        figure_height = height_in / (1 - top_margin - bottom_margin)
        # set new height
        if figure_height > init_height:
            fig.set_figheight(figure_height)

        # set plot title
        plt.title("MAP = {0:.2f}%".format(map*100) +" - "+ label + " label", fontsize=14)
        # set axis titles
        # plt.xlabel('classes')
        plt.xlabel("Average Precision", fontsize='large')
        plt.ylabel("Queries", fontsize='large')
        # adjust size of window
        fig.tight_layout()

        # save the plot
        # fig.savefig("mAP.png")  

        encoded = fig_to_base64(fig)
        imgsrc = 'data:image/png;base64, {}'.format(encoded.decode('utf-8'))

        return render_template('evaluation.html', imgsrc = imgsrc)
    else:
        return render_template('evaluation.html')

# read file
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

# caculate AP
def caculate_AP(result, dataLabel):
    p = 0.0  # precision
    countRel = 0  # count relevant
    ap = 0.0  # average precision
    for i, n in enumerate(dataLabel):
        if dataLabel[i]:
            try:
                #Relevant
                result.index(dataLabel[i])
                countRel += 1
            except ValueError:
                #Irrelevant
                a = 1
            p += countRel/(i+1)

    ap = p/len(dataLabel)
    print("===> AP: " + str(ap))
    return ap

# caculate Average (mean)
def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t

    avg = sum_num / len(num)
    return avg

#encode base64 img
def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)

    return base64.b64encode(img.getvalue())

def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])


if __name__ == "__main__":
    app.run("0.0.0.0")
