import xml.etree.ElementTree as XML

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

def buildStyleAttributes(attribs):
    style = ""
    for a in attribs:
        style = style + a + ";"
    return style

def showXML(e):
    XML.dump(e)
    
def addComponent(parent, component, data):
    c = XML.SubElement(parent,component)
    attributes = []
    attributes.append("left:" + str(data[0]))
    attributes.append("top:" + str(data[1]))
    attributes.append("right:" + str(data[2]))
    attributes.append("bottom:" + str(data[3]))
    attributes.append("width:" + str(data[4]))
    attributes.append("height:" + str(data[5]))
    style = buildStyleAttributes(attributes)
    c.set("style",style)
    return parent
    
def createDocument():
    doc = XML.Element("div")
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

