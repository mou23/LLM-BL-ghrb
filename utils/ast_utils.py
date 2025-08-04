import typed_ast.ast27 as ast

def parse_python_file(sha,file_content):
    class_names = []
    function_names = []
    try:
        tree = ast.parse(file_content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_names.append(node.name)
            
            elif isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
    except:
        print(f"{sha} parse error")
    return ' '.join(class_names),' '.join(function_names)
