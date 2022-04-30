from flask import Blueprint, render_template
from flask import current_app


mod = Blueprint('pages', __name__, url_prefix='/')


@mod.route('/', methods=['GET'])
def main():
    return current_app.send_static_file('index.html')


@mod.route('/dev_mode', methods=['GET'])
def main_dev_page():
    return render_template('analyze_image_dev_mode_page.html')

