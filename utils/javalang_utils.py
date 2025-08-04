import javalang

def parse_java_file(sha,file_content):
    # print(f"parsing file: {sha}")
    class_names = []
    method_names = []
    try:
        tree = javalang.parse.parse(file_content)
        for path, node in tree:
            if isinstance(node, javalang.tree.ClassDeclaration):
                class_names.append(node.name)
            elif isinstance(node, javalang.tree.MethodDeclaration):
                method_names.append(node.name)
    except:
        print(f"{sha} parse error")
    return ' '.join(class_names), ' '.join(method_names)
    
    