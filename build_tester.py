import xml.etree.ElementTree as XML
import page_builder as pb

c = pb.createDocument()
c = pb.addComponent(c,"input",["0","1","2","3","4","5"])
jsfiles = ["app.js","yui.js","bootstrap.js"]
tags = pb.buildScriptTags(jsfiles)
for x in tags.iter():
    XML.dump(x)

