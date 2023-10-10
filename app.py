import time
import argparse
from easyphoto.easyphoto_ui import on_ui_tabs
from easyphoto.easyphoto_utils import reload_javascript
from easyphoto.api import easyphoto_infer_forward_api, easyphoto_train_forward_api


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        help="launch gradio with given server port, you need root/admin rights for ports < 1024, defaults to 7860 if available",
        default=None
    )
    parser.add_argument(
        "--share",
        action='store_true',
        help="use share=True for gradio and make the UI accessible through their site"
    )
    parser.add_argument(
        "--listen",
        action='store_true',
        help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests"
    )
    args = parser.parse_args()

    # load javascript
    reload_javascript()

    # create ui
    easyphoto = on_ui_tabs()

    # launch gradio
    app, _, _ = easyphoto.queue(status_update_rate=1).launch(
        server_name="0.0.0.0" if args.listen else "127.0.0.1",
        server_port=args.port,
        share=args.share,
        prevent_thread_lock=True
    )
    
    # launch api
    easyphoto_infer_forward_api(None, app)
    easyphoto_train_forward_api(None, app)
    
    # not close the python
    while True:
        time.sleep(5)