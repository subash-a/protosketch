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

def getDropDownStructure():
    div = XML.Element("div",{"class":"dropdown"})
    a_main = XML.SubElement(div,"a",{"class":"dropdown-toggle","data-toggle":"dropdown","href":"#"})
    a_main.text = "Menu"
    ul = XML.SubElement(div,"ul",{"class":"dropdown-menu","role":"menu"})
    li_1 = XML.SubElement(ul,"li")
    a_1 = XML.SubElement(li_1,"a",{"href":"#"})
    a_1.text = "link1"
    li_2 = XML.SubElement(ul,"li")
    a_2 = XML.SubElement(li_2,"a",{"href":"#"})
    a_2.text ="link2"
    li_3 = XML.SubElement(ul,"li")
    a_3 = XML.SubElement(li_3,"a",{"href":"#"})
    a_3.text = "link3"
    return div

def getTabStructure():
    return None

def getSliderStructure():
    return None

def getComponent(parent, component):
    if(component == "input_box"):
        v = XML.SubElement(parent,"input")
        v.set("type","text")
        return v
    elif(component == "radio_button"):
        v = XML.SubElement(parent,"input")
        v.set("type","radio")
        return v
    elif(component == "dropdown"):
        v = XML.SubElement(parent,"button")
        v.set("type","button")
        v.text = "Click"
        return v
    elif(component == "check_box"):
        v = XML.SubElement(parent,"input")
        v.set("type","checkbox")
        return v
    elif(component == "slider"):
        v = XML.SubElement(parent,"div")
        v.set("class","slider")
        v.text = " "
        return v
    elif(component == "tab"):
        v = XML.SubElement(parent,"div")
        v.set("class","tab")
        v.text = " "
        return v
    elif(component == "button"):
        v = XML.SubElement(parent,"div")
        v.append(getDropDownStructure())
        return v
    else:
        v = XML.SubElement(parent,"div")
        v.set("class","none")
        v.text = " "
        return vi

def addComponent(parent, component, coords, width, height):
    xml_comp = getComponent(parent,component)
    style_attributes = []
    style_attributes.append("left:" + str(coords[0]))
    style_attributes.append("top:" + str(coords[1]))
#    style_attributes.append("width:" + str(width))
#    style_attributes.append("height:" + str(height))
    style_attributes.append("position: absolute")
    style = buildStyleAttributes(style_attributes)
    xml_comp.set("style",style)
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

