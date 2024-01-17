
Open_Source= ['"The EMBO journal"', '"EMBO molecular medicine"', '"eLife"', '"Science advances"', 
              '"Cell reports"', '"Nature communications"', '"Scientific reports"', '"PloS one"',
              '"Proceedings of the National Academy of Sciences of the United States of America"', 
              '"Nucleic acids research"', '"The Journal of cell biology"', '"iScience"', '"Stem cell reports"', 
              '"Cell Genomics"', '"PLoS genetics"', '"Genome biology"', '"PLoS computational biology"', 
              '"PLoS biology"', '"EMBO reports"', '"The Journal of clinical investigation"', '"BMC biology"', 
              '"BMC genomics"', '"BMC cancer"', '"PLoS Med"']

CommonJounrals = ['Blood', 'Nucleic acids research', 'PloS one', 'The Journal of cell biology', 'PLoS genetics', 
                  'PLoS computational biology', 'PLoS biology','The Journal of clinical investigation', 'Development (Cambridge, England)',
                  'Cancer discovery', 'Cancer research', 'The Journal of experimental medicine' ] 


JournalOA = ['"Nature"', '"Autophagy"', '"Immunity"', '"Nature medicine"', '"Cancer discovery"',
            '"Nature genetics"', '"Cell"', '"Cell stem cell"', '"Cancer cell"', '"Science"', '"Nature cancer"', 
            '"Science immunology"', '"Gastroenterology"', '"Science translational medicine"', 
            '"Cell death and differentiation"', '"Nature cell biology"', '"Molecular cell"', '"Cell metabolism"', 
            '"Oncogene"', '"Cell host microbe"', '"Nature immunology"', '"Blood"', '"Cancer research"', 
            '"Science signaling"', '"nature aging"', '"Current biology : CB"', '"Nature neuroscience"', 
            '"Nature metabolism"', '"Neuron"', '"The Journal of experimental medicine"', 
             '"Developmental cell"', '"Gut"', '"Development (Cambridge, England)"', '"The Journal of biological chemistry"',
             '"Lancet (London, England)"', '"The Lancet. Oncology"', '"The New England journal of medicine"',
              '"Journal of clinical oncology : official journal of the American Society of Clinical Oncology"',
              '"nature structural & molecular biology"', '"Annals of epidemiology"', '"Emerging infectious diseases"',
              '"International journal of infectious diseases : IJID : official publication of the International Society for Infectious Diseases"',
              '"The Journal of infection"', '"The Lancet. Microbe"', '"BMC infectious diseases"', '"BMC immunology"', '"The lancet. HIV"', 
              '"Clinical infectious diseases : an official publication of the Infectious Diseases Society of America"',
              '"Journal of hepatology"', '"Allergy"'] 
pubmedjournals = ['"Nucleic acids research"', '"Nature reviews. Drug discovery"', '"Frontiers in immunology"',
            '"Journal of inflammation (London, England)"', '"Molecular Oncology"', '"The FEBS journal"',
            '"Cell Mol Gastroenterol Hepatol."', '"Frontiers in genetics"', '"Cancers"', '"Frontiers in oncology"']


JOURNALS = list(set(Open_Source + JournalOA + pubmedjournals)) 