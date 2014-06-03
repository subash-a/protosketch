import xml.etree.ElementTree as XML
import page_builder as pb

c = pb.createDocument()
c = pb.addComponent(c,"input",["0","1","2","3","4","5"])
jsfiles = ["utils/js/jquery-1.7.1.js","utils/js/bootstrap.js"]
cssfiles = ["utils/css/bootstrap.css","utils/css/prototype.css"]
h = pb.createHTMLPage(c,jsfiles,cssfiles)
et = XML.ElementTree()
et.write(XML.tostring(h),"single.html")
pb.buildStyleAttributes(["left=200","right=300"])



