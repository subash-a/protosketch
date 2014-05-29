import xml.etree.ElementTree as XML

    
def addComponent(parent, component, data):
    c = XML.SubElement(parent,component)
    c.set("left", data[0])
    c.set("top", data[1])
    c.set("right", data[2])
    c.set("bottom", data[3])
    c.set("width", data[4])
    c.set("height", data[5])
    return parent
    
def createDocument():
    doc = XML.Element("screen")
    comment = XML.Comment("Elements found in the prototype sketch")
    doc.append(comment)
    return doc

def createHTMLPage(xml_template,js_src,css):
    html = XML.Element("html")
    head = XML.SubElement(html,"head")
    body = XML.SubElement(html,"body")
    body.append(xml_template)
    return html

def buildScriptTags(js_list):
    tags = XML.Element("tags")
    for x in js_list:
        script = XML.Element("script")
        script.set("src",x)
        tags.append(script)
    return tags
        
