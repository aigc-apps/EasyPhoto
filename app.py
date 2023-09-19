from easyphoto.easyphoto_ui import on_ui_tabs
from easyphoto.easyphoto_utils import reload_javascript

if __name__ == "__main__": 
    reload_javascript()
    easyphoto = on_ui_tabs()
    easyphoto.queue(status_update_rate=1).launch()