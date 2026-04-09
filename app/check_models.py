import traceback
import engine

if __name__ == '__main__':
    try:
        engine.init_models()
        print('first_gate:', 'loaded' if engine.first_gate is not None else 'NOT LOADED')
        print('second_gate:', 'loaded' if engine.second_gate is not None else 'NOT LOADED')
        print('location_model:', 'loaded' if engine.location_model is not None else 'NOT LOADED')
        print('severity_model:', 'loaded' if engine.severity_model is not None else 'NOT LOADED')
        print('cat_list:', 'loaded' if engine.cat_list is not None else 'NOT LOADED')
    except Exception:
        traceback.print_exc()
