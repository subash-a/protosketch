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

def createHTMLPage(xml_template,js_list,css_list):
    html = XML.Element("html")
    head = XML.SubElement(html,"head")
    body = XML.SubElement(html,"body")
    css_list = buildStyleTags(css_list)
    for c in css_list:
        head.append(c)
    body.append(xml_template)
    js_list = buildScriptTags(js_list)
    for j in js_list:
        body.append(j)
    return html

def buildScriptTags(js_list):
    tags = XML.Element("tags")
    for x in js_list:
        script = XML.Element("script")
        script.set("src",x)
        script.text = " "
        tags.append(script)
    return tags
        
def buildStyleTags(css_list):
    tags = XML.Element("tags")
    for x in css_list:
        link = XML.Element("link")
        link.set("rel", "stylesheet")
        link.set("type", "text/css")
        link.set("src", x)
        tags.append(link)
    return tags
