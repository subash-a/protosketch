import xml.etree.ElementTree as XML
import page_builder as pb

x = XML.Element("html")
y = XML.SubElement(x,"head")
z = XML.SubElement(x,"body")
a = XML.SubElement(z,"div",{"name":"girl"})
a.text = " "
#XML.SubElement(z,"input",{"type":"radio"})
XML.SubElement(z,"div",{"name":"boy"})
ET = XML.ElementTree(x)
ET.write("output/testpage.html")
