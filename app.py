import time

from easyphoto.easyphoto_ui import on_ui_tabs
from easyphoto.easyphoto_utils import reload_javascript
from easyphoto.api import easyphoto_infer_forward_api, easyphoto_train_forward_api


if __name__ == "__main__": 
    # load javascript
    reload_javascript()

    # create ui
    easyphoto = on_ui_tabs()

    # launch gradio
    app, _, _ = easyphoto.queue(status_update_rate=1).launch(prevent_thread_lock=True)
    easyphoto_infer_forward_api(None, app)
    easyphoto_train_forward_api(None, app)
    
    # not close the python
    while True:
        time.sleep(5)