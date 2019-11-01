"""
Script to:
Returns:

Author:     Romaric & Kewin
"""
from SPARQLWrapper import SPARQLWrapper, JSON
import pickle
import os.path
import pandas as pd


def get_all_sample_uris(limit=0):
    """Extract sample URIs from the http://nvme1.wurnet.nl:7200 prokaryote
    SPARQL repository

    :return: list of gbol prokaryote sample URIs
    """
    sample_uri_list = []

    query = """
    #list of all samples
    PREFIX gbol: <http://gbol.life/0.1/>
    SELECT ?sample
    WHERE { 
        ?sample ?p gbol:Sample .
    } """

    if limit > 0:
        query += " limit " + str(limit)

    print("Getting sample URIs")

    sparql = SPARQLWrapper("http://nvme1.wurnet.nl:7200/repositories/Prokaryotes")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        sample_uri_list.append(result["sample"]["value"])
        # print(result["sample"]["value"])
    return sample_uri_list


def get_sample_info(sample_uri):
    """Return parsed sample organism name, protein domains and amount of domains
    from the http://nvme1.wurnet.nl:7200 prokaryote SPARQL repository

    :param sample_uri: string
    :return: list of lists of [[Species, domain, domain count], -->]
    """
    sample_info = []

    query = """
    #For a sample in VALUE retrieve the organisms, pfam protein domains (accession no) and their count
    PREFIX gbol: <http://gbol.life/0.1/>
    SELECT ?name ?accession (COUNT(?accession) AS ?accession_count) 
    WHERE {

        VALUES ?sample {<%s>}
        VALUES ?db {<http://gbol.life/0.1/db/pfam>}
        ?sample a gbol:Sample .
        ?contigs gbol:sample ?sample ;
                 gbol:feature ?gene ;
                 gbol:description ?description ;
                 gbol:organism ?organism .
        ?organism gbol:scientificName ?name .
        ?gene gbol:transcript ?transcript .
        ?transcript gbol:feature ?CDS.
        ?CDS gbol:protein ?protein.
        ?protein gbol:feature ?domain.
        ?domain gbol:xref ?xref.
        ?xref gbol:db ?db .
        ?xref gbol:accession ?accession
    } GROUP BY ?name ?accession 
    """ % sample_uri

    print("Getting sample information of", sample_uri)

    sparql = SPARQLWrapper("http://nvme1.wurnet.nl:7200/repositories/Prokaryotes")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        sample_info.append([' '.join(result["name"]["value"].split()[0:2]) + "_#_" + sample_uri.split("/")[-2],
                            result["accession"]["value"],
                            int(result["accession_count"]["value"])])

    # http://gbol.life/0.1/GCA_002128005/sample  # example sample URI
    return sample_info


def get_all_samples_info(limit=0):
    """Retrieve sample info of all samples and store as pickle

    :param limit: number of samples used from the repository, default=0.
    :return: all_sample_info: nested dict of {species: {domain: count, -->}}
    all_sample_info is also stored as pickle named all_sample_info_dict.pickle{limit}
    """
    sample_uri_list = get_all_sample_uris(limit)
    all_samples_info_dict = {}
    for sample_uri in sample_uri_list:
        sample_info = get_sample_info(sample_uri)
        for domain in sample_info:
            if domain[0] not in all_samples_info_dict:
                all_samples_info_dict[domain[0]] = {domain[1]: domain[2]}
            elif domain[1] not in all_samples_info_dict[domain[0]]:  # species already in dict
                all_samples_info_dict[domain[0]][domain[1]] = domain[2]
    return all_samples_info_dict


def dict_to_DataFrame(data):
    # Convert nested dictionary to pandas DataFrame with rows = species and col = domains
    df = pd.DataFrame(data).transpose()
    # print(df.head())

    # Replace row names species_sample with species
    old_name_list = []
    new_name_list = []
    for name in data:
        old_name_list.append(name)
        new_name_list.append(name.split('_')[0])
    for nr in range(len(new_name_list)):
        df.rename(index={old_name_list[nr]: new_name_list[nr]}, inplace=True)

    # Replace NaNs by 0
    df = df.fillna(0)
    # print(df.head())
    return df


def get_data_info(all_samples_info_df):
    # Get amount of domains
    for row in range(len(all_samples_info_df)):
        non_zero_count = 0
        for domain in all_samples_info_df.iloc[row]:
            if domain != 0:
                non_zero_count += 1
        print(f"{all_samples_info_df.iloc[row].name} "
              f"has {100 * non_zero_count / len(all_samples_info_df.columns):.0f}% "
              f"domains ({non_zero_count}/{len(all_samples_info_df.columns)})")

    print(f"\nThere are {len(all_samples_info_df)} species of which "
          f"{len(all_samples_info_df.index.unique())} unique")
    print(f'There are {len(all_samples_info_df.columns)} domains')


def main(limit=0):
    # If pickle with specified limit doesn't exist: make it and store as pickle
    # else load pickle
    if not os.path.exists(f'/home/kewin/PycharmProjects/Capita_Selecta/all_samples_info_df{limit}.pickle'):
        print(f'File with limit {limit} does not exist yet, making it...')
        data_dict = get_all_samples_info(limit)
        all_samples_info_df = dict_to_DataFrame(data_dict)
        print('\nDf made, storing it as pickle')
        with open(f"all_samples_info_df{limit}.pickle", "wb") as f:  # f is temp variable
            pickle.dump(all_samples_info_df, f, pickle.HIGHEST_PROTOCOL)
        print("\nDone")
    else:
        print(f'\nFile with limit {limit} exists, loading pickle...')
        with open(f"all_samples_info_df{limit}.pickle", 'rb') as pickle_in:
            all_samples_info_df = pickle.load(pickle_in)
        print('Done\n')

    get_data_info(all_samples_info_df)



if __name__ == "__main__":
    main(limit=0)
