import json
import os
import subprocess
import re
import codecs

def write_file(path,content):
    with open(path,'w') as f:
        f.write(content)

def parseAst(path):
    p = subprocess.Popen(f"/deepo_data/gumtree-3.0.0/bin/gumtree parse {path}",shell=True,text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if stderr:
        print(f"parser error. {path} - {stderr}")
        return json.dumps('')
    else:
        return stdout
    
def parseAstJson(path):
    p = subprocess.Popen(f"/deepo_data/gumtree-3.0.0/bin/gumtree parse {path} -f json",shell=True,text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if stderr:
        print(f"parser error. {path} - {stderr}")
        return json.dumps('')
    else:
        return stdout
    
class ASTNode(object):
    def __init__(self, temp, children):
        self.type = temp["type"]
        # 值
        self.label = temp.get("label")
        self.pos = int(temp["pos"])
        self.length = int(temp["length"])
        self.children = children
        self.beforeID = temp["id"]
        self.afterID = None
        self.parent = None
        self.operation = None

    def __repr__(self):
        return f"type: {self.type}, label: {self.label}, pos: {self.pos}, length: {self.length}, id: {self.beforeID}"


class AST(object):

    def __init__(self, ASTJSON):
        self.ASTNodeList = []
        AST = json.loads(ASTJSON)
        self.id = 0
        self.constructTreeByJSON(AST["root"])
        self.addNodeParent(self.ASTNodeList[-1], None)

    # 构建树，
    def constructTreeByJSON(self, subASTTree):
        children = []
        if subASTTree["children"] != []:
            for i in subASTTree["children"]:
                children.append(self.constructTreeByJSON(i))
        subASTTree["id"] = self.id
        self.id = self.id + 1
        # print(subASTTree)
        ASTNodeObj = ASTNode(subASTTree, children)
        self.ASTNodeList.append(ASTNodeObj)
        return ASTNodeObj

    def addNodeParent(self, subTree, parent):
        subTree.parent = parent
        for i in subTree.children:
            self.addNodeParent(i, subTree)

    def astToJson(self, i=-1):
        def astToDict(node):
            tempDict = {"type": node.type, "pos": str(node.pos), "length": str(node.length)}
            if node.label != None:
                tempDict["label"] = node.label
            childrenList = []
            # print(node.children)
            for j in node.children:
                # print("fuck")
                childrenList.append(astToDict(j))
            tempDict["children"] = childrenList
            return tempDict

        return json.dumps(astToDict(self.ASTNodeList[i]))

    def getNodeByID(self, nodeID) -> ASTNode:
        return self.ASTNodeList[nodeID]

    def getHeadNode(self):
        return self.ASTNodeList[-1]

    def insertNode(self, parentID, index, node):
        parentNode = self.getNodeByID(parentID)
        parentNode.children.insert(index, node)
        node.parent = parentNode

class ParserCorpusMethodLevelGranularity:
    def __init__(self,path) -> None:
        self.path = path
        with codecs.open(path,'r',errors='ignore') as f:
            self.filecontent = f.read()
        self.packageName = ''
        self.methodname = []
        
    def parseSourceCode(self):
        p = subprocess.Popen(f"java -jar ./lib/gumtree.jar parse {self.path} -f json",shell=True,text=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if stderr:
            print(f"{self.path} - {stderr}")
            return None
        else:
            try:
                tree = AST(stdout)
            except:
                print(f"{self.path} - {stderr}")
                return None
            return self.exploreSourceCode(tree.getHeadNode())
    
    def exploreSourceCode(self,tree):
        for node in tree.children:
            if node.type == 'PackageDeclaration':
                for i in node.children:
                    if i.type == "QualifiedName":
                        self.packageName = i.label
                        break
            if node.type == 'TypeDeclaration':
                self.exploreClassContents(node,'')
    
    def exploreClassContents(self,node,prefixClass):
        Modifier = []
        className = ''
        fullClassName = ''
        for i in node.children:
            if i.type == 'Modifier':
                Modifier.append(i.label)
            if 'abstract' in Modifier:
                return
            if i.type == 'TYPE_DECLARATION_KIND' and i.label== 'interface':
                return
            if i.type == 'SimpleName':
                className = i.label
                fullClassName = prefixClass + className + "."
            if i.type == 'TypeDeclaration':
                self.exploreClassContents(i,fullClassName)
            if i.type == 'MethodDeclaration':
                self.exploreMethodContents(i,fullClassName)
                
    def exploreMethodContents(self,node,prefixClass):
        Modifier = []
        methodname = ''
        # Javadoc = ''
        Variables = []
        methodcontent = self.filecontent[node.pos:(node.pos+node.length)]
        for i in node.children:
            # if i.type == 'Javadoc':
            #     Javadoc = self.filecontent[i.pos:(i.pos+i.length)]
            if i.type == 'Modifier':
                Modifier.append(i.label)
            if 'virtual' in Modifier:
                return
            if i.type == 'SimpleName':
                methodname = i.label
            if i.type == 'SingleVariableDeclaration':
                for j in i.children:
                    if j.type == 'SimpleType':
                        for k in j.children:
                            if k.type == 'SimpleName':
                                Variables.append(k.label)
                    if j.type == 'PrimitiveType':
                        Variables.append(j.label)
        idMethod = prefixClass + methodname
        methodNameFullPathFinal = idMethod+'('
        if len(Variables)==0:
            methodNameFullPathFinal+=')'
        else:
            for i in Variables[:-1]:
                methodNameFullPathFinal+=i+','            
            methodNameFullPathFinal+=Variables[-1]+')'
        self.methodname.append(methodNameFullPathFinal)

class ParserMethodFile():
    def __init__(self,sha,content,stdout) -> None:
        self.sha = sha
        self.filecontent = content
        self.stdout = stdout
        # self.path = path.replace('(','\(').replace(')','\)').replace('$','\$')
        self.packageName = ''
        self.classname = []
        self.methodname = []
        self.variables = []
        self.methodcontent = ''
        self.comments = []
        # self.imports = []
        
    def parseSourceCode(self):
        try:
            tree = AST(self.stdout)
        except:
            print(f"AST error. sha: {self.sha}")
            return False
        self.exploreSourceCode(tree.getHeadNode())
        return True

    def exploreSourceCode(self,tree):
        for node in tree.children:
            if node.type == 'PackageDeclaration':
                for i in node.children:
                    if i.type == "QualifiedName":
                        self.packageName = i.label
                        break
            if node.type == 'TypeDeclaration':
                prefix = self.packageName+'.' if self.packageName else ''
                self.exploreClassContents(node,prefix)
                
    def exploreClassContents(self,node,prefixClass):
        className = ''
        fullClassName = ''
        for i in node.children:
            # if i.type == 'TYPE_DECLARATION_KIND' and i.label== 'interface':
            #     return
            if i.type == 'SimpleName':
                className = i.label
                fullClassName = prefixClass + className + "."
                self.classname.append(fullClassName[:-1])
            if i.type == 'TypeDeclaration':
                self.exploreClassContents(i,fullClassName)
            if i.type == 'MethodDeclaration':
                self.exploreMethodContents(i,fullClassName)
                
    def exploreMethodContents(self,node,prefixClass):
        self.methodcontent = self.filecontent[node.pos:(node.pos+node.length)]
        for i in node.children:
            if i.type == 'Javadoc':
                self.traverse_varivable_node(i,prefixClass)
            if i.type == 'SimpleName':
                methodname = i.label
                prefixmethd = prefixClass+methodname+'.'
                self.methodname.append(prefixmethd[:-1])
            if i.type == 'SingleVariableDeclaration':
                for j in i.children:
                    if j.type == 'SimpleName':
                        self.variables.append(j.label)
            if i.type == 'Block':
                self.traverse_varivable_node(i,prefixmethd)

    def traverse_varivable_node(self,node,prefixmethd):
        name = ''
        # 'FieldAccess', 'ArrayAccess','Assignment','ReturnStatement','InfixExpression','PostfixExpression','CastExpression','MethodInvocation','METHOD_INVOCATION_RECEIVER','METHOD_INVOCATION_ARGUMENTS','InstanceofExpression','ClassInstanceCreation','VariableDeclarationFragment','SingleVariableDeclaration'
        if node.type in ['VariableDeclarationFragment','SingleVariableDeclaration']:
            for i in node.children:
                if i.type == 'SimpleName':
                    self.variables.append(i.label)
        # if node.type == 'ClassInstanceCreation':
        #     for i in node.children:
        #         if i.type == 'SimpleType':
        #             for j in i.children:
        #                 if j.type == 'SimpleName':
        #                     name = j.label
        #                     self.classname.append(name)
        #                     break
        #             break
        elif node.type == 'MethodDeclaration':
            for i in node.children:
                if i.type == 'SimpleName':
                    name = i.label
                    self.methodname.append(prefixmethd+name)
                    break
        if node.type == 'TextElement':
            if node.label:
                self.comments.append(node.label)
        if node.type == 'TagElement':
            for i in node.children:
                if i.type == 'SimpleName':
                    self.comments.append(i.label)
        if node.children:
            for i in node.children:
                self.traverse_varivable_node(i,prefixmethd)

if __name__ == '__main__':
    print(parseAstJson('/deepo_data/experiments/datasets/Apache/HTTPCLIENT/sources/HTTPCLIENT_4_0/httpclient/src/main/java/org/apache/http/auth/AuthSchemeFactory.java'))
    # from git_utils import cat_file_blob
    # from unqlite import UnQLite
    # UNQLITE_OPEN_READONLY = 0x00000001
    # gitrepo = '/deepo_data/baselines/Bench4BL/data/Spring/BATCH/gitrepo'
    # # file_path = 'spring-batch-core/src/main/java/org/springframework/batch/core/DefaultJobKeyGenerator#generateKey\(JobParameters\).java'
    # sha = '71cc0b315b5802287e67a800db1c52410e168125'
    # db = UnQLite('/deepo_data/codes/BMBL/results/BATCH/ast_db',flags = UNQLITE_OPEN_READONLY)
    # ast_json = db[sha]
    # content = cat_file_blob(gitrepo,sha)
    # parser = ParserMethodFile(sha,content,ast_json)
    # # result = parser.parseAst()
    # # result = json.loads(result)
    # # print(result)
    # flag = parser.parseSourceCode()
    # # print(parser.filecontent)
    # if flag:
    #     print(parser.packageName)
    #     print(parser.classname)
    #     print(parser.methodname)
    #     print(parser.variables)
    #     print(parser.comments)
    #     print(parser.methodcontent)
    