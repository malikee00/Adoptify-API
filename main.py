import pandas as pd

from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from db import connect_tcp_socket
from dotenv import load_dotenv

dotenv_path = "./.env"
load_dotenv(dotenv_path=dotenv_path)

app = FastAPI()


@app.post("/api/pet-recommendations")
async def pet_recommendations(petId: int, recomType: str):
    # db = connect_tcp_socket()
    # print("🍔🍔🔥 connection success")

    # data = pd.read_sql()
    data = pd.read_csv("./data/Dataset Table - Dataset Adoptify.csv")
    # data_original = data.copy()

    # jumlah_jenis_hewan = data.groupby("Jenis").size().reset_index(name="Jumlah")
    data["Kontak"] = data["Kontak"].str.replace("'", "")
    data.rename(columns={"ID": "UID"}, inplace=True)
    # RAS

    tf = TfidfVectorizer()
    tf.fit(data["Ras"])
    # tf.get_feature_names_out()

    tfidf_matrix_ras = tf.fit_transform(data["Ras"])
    # tfidf_matrix_ras.shape

    # tfidf_matrix_ras.todense()

    # pd.DataFrame(
    #     tfidf_matrix_ras.todense(), columns=tf.get_feature_names_out(), index=data.UID
    # ).sample(20, axis=1).sample(10, axis=0)

    cosine_sim_ras = cosine_similarity(tfidf_matrix_ras)
    #     cosine_sim_ras

    cosine_sim_df_ras = pd.DataFrame(
        cosine_sim_ras, index=data["UID"], columns=data["UID"]
    )

    #     print('shape :', cosine_sim_df_ras.shape)

    # cosine_sim_df_ras.sample(5, axis = 1).sample(10, axis = 0)

    def ras_hewan_recommendations(
        UID, similarity_data=cosine_sim_df_ras, items=data, k=5
    ):
        index = similarity_data.loc[:, UID].to_numpy().argpartition(range(-1, -k, -1))
        closest = similarity_data.columns[index[-1 : -(k + 2) : -1]]
        closest = closest.drop(UID, errors="ignore")
        return pd.DataFrame(closest).merge(items).head(k)

    #     data[data.UID.eq(3028)]
    #     ras_hewan_recommendations(3028)

    # Kesehatan
    tf = TfidfVectorizer()
    tf.fit(data["Kesehatan"])

    #     tf.get_feature_names_out()

    tfidf_matrix_kesehatan = tf.fit_transform(data["Kesehatan"])
    #     tfidf_matrix_kesehatan.shape
    #     tfidf_matrix_kesehatan.todense()

    #     pd.DataFrame(
    #     tfidf_matrix_kesehatan.todense(),
    #     columns = tf.get_feature_names_out(),
    #     index = data.UID).sample(5, axis = 1). sample(10, axis = 0)

    cosine_sim_kesehatan = cosine_similarity(tfidf_matrix_kesehatan)
    cosine_sim_df_kesehatan = pd.DataFrame(
        cosine_sim_kesehatan, index=data["UID"], columns=data["UID"]
    )
    #     print('shape :', cosine_sim_df_kesehatan.shape)
    #     cosine_sim_df_kesehatan.sample(5, axis = 1).sample(10, axis = 0)

    def kesehatan_hewan_recommendations(
        UID,
        similarity_data=cosine_sim_df_kesehatan,
        items=data,
        k=5,
    ):
        index = similarity_data.loc[:, UID].to_numpy().argpartition(range(-1, -k, -1))
        closest = similarity_data.columns[index[-1 : -(k + 2) : -1]]
        closest = closest.drop(UID, errors="ignore")
        return pd.DataFrame(closest).merge(items).head(k)

    #     data[data.UID.eq(1078)]
    #     kesehatan_hewan_recommendations(4002)

    # JENIS
    tf = TfidfVectorizer()
    tf.fit(data["Jenis"])
    tf.get_feature_names_out()

    tfidf_matrix_jenis = tf.fit_transform(data["Jenis"])
    #     tfidf_matrix_jenis.shape

    #     tfidf_matrix_jenis.todense()
    #     pd.DataFrame(
    #     tfidf_matrix_jenis.todense(),
    #     columns = tf.get_feature_names_out(),
    #     index = data.UID).sample(6, axis = 1). sample(10, axis = 0)

    cosine_sim_jenis = cosine_similarity(tfidf_matrix_jenis)
    cosine_sim_df_jenis = pd.DataFrame(
        cosine_sim_jenis, index=data["UID"], columns=data["UID"]
    )
    #     print('shape :', cosine_sim_df_jenis.shape)

    #     cosine_sim_df_jenis.sample(10, axis = 1).sample(10, axis = 0)

    def jenis_hewan_recommendations(
        UID, similarity_data=cosine_sim_df_jenis, items=data, k=5
    ):
        index = similarity_data.loc[:, UID].to_numpy().argpartition(range(-1, -k, -1))
        closest = similarity_data.columns[index[-1 : -(k + 2) : -1]]
        closest = closest.drop(UID, errors="ignore")
        return pd.DataFrame(closest).merge(items).head(k)

    # MEAN RAS KESEHATAN
    mean_data = (cosine_sim_df_kesehatan + cosine_sim_df_ras) / 2
    #     mean_data.sample(5, axis = 1).sample(10, axis = 0)
    # # mean_data[4002]

    def mean_kesehatan_ras_recommendation(
        UID, similarity_data=mean_data, items=data, k=10
    ):
        index = similarity_data.loc[:, UID].to_numpy().argpartition(range(-1, -k, -1))
        closest = similarity_data.columns[index[-1 : -(k + 2) : -1]]
        closest = closest.drop(UID, errors="ignore")
        return pd.DataFrame(closest).merge(items).head(k)

    if recomType == "ALL":
        df_result = mean_kesehatan_ras_recommendation(petId)
    if recomType == "RAS":
        df_result = ras_hewan_recommendations(petId)
    if recomType == "KESEHATAN":
        df_result = kesehatan_hewan_recommendations(petId)
    if recomType == "JENIS":
        df_result = jenis_hewan_recommendations(petId)
    # print("🍔🍔 ==>", result)

    return {
        "status": 200,
        "msg": "Success Generate Recommendations",
        "data": df_result.to_dict("records"),
    }
