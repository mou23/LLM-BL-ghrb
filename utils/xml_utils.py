import xml.etree.cElementTree as ET

def read_report(report):
    root = ET.parse(report).getroot()
    bugs = list(root.iter('bug'))
    bugitems = {}
    for bug in bugs:
        id = bug.get('id')
        bugitems[id] = {}
        bug_info_element = bug.find('buginformation')
        summary = bug_info_element.find('summary').text
        description = bug_info_element.find('description').text
        bugitems[id]['text'] = summary+ ' '+ description if description else summary
        fixed_files_element = bug.find('fixedFiles')
        files = fixed_files_element.findall('file')
        bugitems[id]['fixedFiles'] = [i.text for i in files]
    return bugitems