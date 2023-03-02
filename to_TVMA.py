import re
import os
import xml.etree.cElementTree as ET
regex_float_pattern = r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'


def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def build_tree(xgtree, base_xml_element, var_indices):
    parent_element_dict = {'0':base_xml_element}
    pos_dict = {'0':'s'}
    for line in xgtree.split('\n'):
        if not line: continue
        if ':leaf=' in line:
            #leaf node
            result = re.match(r'(\t*)(\d+):leaf=({0})$'.format(regex_float_pattern), line)
            if not result:
                print(line)
            depth = result.group(1).count('\t')
            inode = result.group(2)
            res = result.group(3)
            node_elementTree = ET.SubElement(parent_element_dict[inode], "Node", pos=str(pos_dict[inode]),
                                             depth=str(depth), NCoef="0", IVar="-1", Cut="0.0e+00", cType="1", res=str(res), rms="0.0e+00", purity="0.0e+00", nType="-99")
        else:
            #\t\t3:[var_topcand_mass<138.19] yes=7,no=8,missing=7
            result = re.match(r'(\t*)([0-9]+):\[(?P<var>.+)<(?P<cut>{0})\]\syes=(?P<yes>\d+),no=(?P<no>\d+)'.format(regex_float_pattern),line)
            if not result:
                print(line)
            depth = result.group(1).count('\t')
            inode = result.group(2)
            var = result.group('var')
            cut = result.group('cut')
            lnode = result.group('yes')
            rnode = result.group('no')
            pos_dict[lnode] = 'l'
            pos_dict[rnode] = 'r'
            node_elementTree = ET.SubElement(parent_element_dict[inode], "Node", pos=str(pos_dict[inode]),
                                             depth=str(depth), NCoef="0", IVar=str(var_indices[var]), Cut=str(cut),
                                             cType="1", res="0.0e+00", rms="0.0e+00", purity="0.0e+00", nType="0")
            parent_element_dict[lnode] = node_elementTree
            parent_element_dict[rnode] = node_elementTree
            
def convert_model(model, itree, input_variables, output_xml):
    NTrees = len(model)
    var_list = input_variables
    var_indices = {}
    
    # <MethodSetup>
    # MethodSetup = ET.Element("MethodSetup", Method="BDT::BDT")
    BinaryTree = ET.Element("BinaryTree", type="DecisionTree", boostWeight="1.0e+00", itree=str(itree))

    # <Variables>
 #   Variables = ET.SubElement(BinaryTree, "Variables", NVar=str(len(var_list)))
    for ind, val in enumerate(var_list):
        name = val[0]
        var_type = val[1]
        var_indices[name] = ind
        # #Variable = ET.SubElement(Variables, "Variable", VarIndex=str(ind), Type=val[1], 
        #     Expression=name, Label=name, Title=name, Unit="", Internal=name, 
        #     Min="0.0e+00", Max="0.0e+00")

    # # <GeneralInfo>
    # GeneralInfo = ET.SubElement(MethodSetup, "GeneralInfo")
    # Info_Creator = ET.SubElement(GeneralInfo, "Info", name="Creator", value="xgboost2TMVA")
    # Info_AnalysisType = ET.SubElement(GeneralInfo, "Info", name="AnalysisType", value="Classification")

    # # <Options>
    # Options = ET.SubElement(MethodSetup, "Options")
    # Option_NodePurityLimit = ET.SubElement(Options, "Option", name="NodePurityLimit", modified="No").text = "5.00e-01"
    # Option_BoostType = ET.SubElement(Options, "Option", name="BoostType", modified="Yes").text = "Grad"
    
    # <Weights>
    #Weights = ET.SubElement(MethodSetup, "Weights", NTrees=str(NTrees), AnalysisType="1")
    
    for itree in range(NTrees):
        build_tree(model[itree], BinaryTree, var_indices)
        
    tree = ET.ElementTree(BinaryTree)

    indent(BinaryTree)
    op = ET.tostring(BinaryTree)
   
    tree.write(output_xml) 
    
    # format it with 'xmllint --format'
    
# example
# bst = xgb.train( param, d_train, num_round, watchlist );
# model = bst.get_dump()
# convert_model(model,input_variables=[('var1','F'),('var2','I')],output_xml='xgboost.xml')
