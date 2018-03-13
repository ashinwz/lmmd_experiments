#chembl 22.1
SELECT a.molregno,a.canonical_smiles,b.chembl_id,c.compound_name,d.assay_id,d.standard_relation,d.published_type,d.published_value,d.published_units,e.tid,f.pref_name,f.organism,f.chembl_id,g.component_id,h.accession 
from compound_structures a 
INNER JOIN molecule_dictionary b ON a.molregno=b.molregno 
INNER JOIN compound_records c ON a.molregno=c.molregno 
INNER JOIN activities d ON a.molregno=d.molregno 
INNER JOIN assays e ON d.assay_id=e.assay_id 
INNER JOIN target_dictionary f on e.tid=f.tid 
INNER JOIN target_components g on f.tid=g.tid 
INNER JOIN component_sequences h on g.component_id=h.component_id 
where d.published_value!='' 
and f.organism<=>"Homo sapiens" 
and d.published_type in("IC50","EC50","Ki","Kd")