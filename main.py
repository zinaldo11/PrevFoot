import streamlit as st
import pandas as pd
import requests
import json
import io
from datetime import datetime
import xlsxwriter
import numpy as np
from scipy.stats import poisson

# Carregar pesos das competições
with open('pesos.json', 'r', encoding='utf-8') as f:
    PESOS = json.load(f)

API_KEY = st.secrets["API_FOOTBALL_KEY"] if "API_FOOTBALL_KEY" in st.secrets else "SUA_API_KEY_AQUI"
API_URL = "https://v3.football.api-sports.io/"

st.set_page_config(page_title="Previsão Estatística Futebol", layout="wide", page_icon="⚽")
st.title("Previsão Estatística de Partidas de Futebol")

# Inicializar session_state
if 'times_a' not in st.session_state:
    st.session_state.times_a = []
if 'times_b' not in st.session_state:
    st.session_state.times_b = []
if 'erro_a' not in st.session_state:
    st.session_state.erro_a = None
if 'erro_b' not in st.session_state:
    st.session_state.erro_b = None
if 'idx_a' not in st.session_state:
    st.session_state.idx_a = None
if 'idx_b' not in st.session_state:
    st.session_state.idx_b = None
if 'temporada_a' not in st.session_state:
    st.session_state.temporada_a = None
if 'temporada_b' not in st.session_state:
    st.session_state.temporada_b = None
if 'jogos_a' not in st.session_state:
    st.session_state.jogos_a = []
if 'jogos_b' not in st.session_state:
    st.session_state.jogos_b = []

# Tabs principais
abas = st.tabs([
    "Seleção dos Times e Temporada",
    "Jogos Analisados",
    "Médias",
    "Estatísticas Previstas",
    "Placar Provável",
    "Odds x Previsão",
    "Exportar Resultado"
])

with abas[0]:
    st.header("Seleção dos Times e Temporada")
    st.markdown("Digite os nomes dos times e clique em 'Buscar Times' para selecionar os times e a temporada:")
    
    headers = {"x-apisports-key": API_KEY}
    
    def buscar_times(nome):
        try:
            url = f"{API_URL}teams?search={nome}"
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("response"):
                return data["response"], None
            else:
                erro_api = data.get("errors", "Nenhum time encontrado.")
                return [], f"Nenhum time encontrado. Erro API: {erro_api}"
        except Exception as e:
            return [], f"Erro na busca: {e}"
    
    def buscar_temporadas(time_id):
        try:
            url = f"{API_URL}teams/seasons?team={time_id}"
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            seasons = data.get("response", [])
            return seasons, None
        except Exception as e:
            return [], f"Erro ao buscar temporadas: {e}"

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mandante")
        time_a_nome = st.text_input("Time A (Mandante)", key="time_a_input")
        
        if st.session_state.erro_a:
            st.error(st.session_state.erro_a)
        elif st.session_state.times_a:
            def format_time_a(i):
                t = st.session_state.times_a[i]['team']
                return f"{t['name']} ({t['country']}, fundado em {t['founded']})"
            st.session_state.idx_a = st.selectbox(
                "Selecione o Time A",
                options=range(len(st.session_state.times_a)),
                format_func=format_time_a,
                key="select_time_a"
            )
            st.image(st.session_state.times_a[st.session_state.idx_a]['team']['logo'], width=60)
            
            # Buscar temporadas disponíveis
            time_a_id = st.session_state.times_a[st.session_state.idx_a]['team']['id']
            seasons_a, erro_seasons_a = buscar_temporadas(time_a_id)
            if erro_seasons_a:
                st.error(erro_seasons_a)
            elif seasons_a:
                latest_season_a = max(seasons_a)
                st.markdown(f"**Temporada mais recente disponível:** {latest_season_a}")
                st.session_state.temporada_a = st.selectbox(
                    "Selecione a temporada do Time A",
                    options=[2020, 2021, 2022, 2023, 2024, 2025],
                    index=[2020, 2021, 2022, 2023, 2024, 2025].index(latest_season_a) if latest_season_a in [2020, 2021, 2022, 2023, 2024, 2025] else 0,
                    key="temporada_a"
                )
            else:
                st.warning("Nenhuma temporada encontrada para o time selecionado.")
        else:
            st.warning("Digite um nome e clique em 'Buscar Times' para encontrar times.")
        
    with col2:
        st.subheader("Visitante")
        time_b_nome = st.text_input("Time B (Visitante)", key="time_b_input")
        
        if st.session_state.erro_b:
            st.error(st.session_state.erro_b)
        elif st.session_state.times_b:
            def format_time_b(i):
                t = st.session_state.times_b[i]['team']
                return f"{t['name']} ({t['country']}, fundado em {t['founded']})"
            st.session_state.idx_b = st.selectbox(
                "Selecione o Time B",
                options=range(len(st.session_state.times_b)),
                format_func=format_time_b,
                key="select_time_b"
            )
            st.image(st.session_state.times_b[st.session_state.idx_b]['team']['logo'], width=60)
            
            # Buscar temporadas disponíveis
            time_b_id = st.session_state.times_b[st.session_state.idx_b]['team']['id']
            seasons_b, erro_seasons_b = buscar_temporadas(time_b_id)
            if erro_seasons_b:
                st.error(erro_seasons_b)
            elif seasons_b:
                latest_season_b = max(seasons_b)
                st.markdown(f"**Temporada mais recente disponível:** {latest_season_b}")
                st.session_state.temporada_b = st.selectbox(
                    "Selecione a temporada do Time B",
                    options=[2020, 2021, 2022, 2023, 2024, 2025],
                    index=[2020, 2021, 2022, 2023, 2024, 2025].index(latest_season_b) if latest_season_b in [2020, 2021, 2022, 2023, 2024, 2025] else 0,
                    key="temporada_b"
                )
            else:
                st.warning("Nenhuma temporada encontrada para o time selecionado.")
        else:
            st.warning("Digite um nome e clique em 'Buscar Times' para encontrar times.")
    
    # Botão único para buscar ambos os times
    buscar = st.button("Buscar Times")
    if buscar:
        if not time_a_nome.strip() or not time_b_nome.strip():
            st.error("Por favor, insira nomes válidos para ambos os times.")
        else:
            with st.spinner("Buscando times..."):
                st.session_state.times_a, st.session_state.erro_a = buscar_times(time_a_nome)
                st.session_state.times_b, st.session_state.erro_b = buscar_times(time_b_nome)

with abas[1]:
    st.header("Jogos Analisados")
    if st.session_state.times_a and st.session_state.times_b and st.session_state.idx_a is not None and st.session_state.idx_b is not None:
        time_a_id = st.session_state.times_a[st.session_state.idx_a]['team']['id']
        time_b_id = st.session_state.times_b[st.session_state.idx_b]['team']['id']
        temporada_a = st.session_state.temporada_a
        temporada_b = st.session_state.temporada_b
        
        def buscar_jogos(time_id, temporada, mando):
            try:
                if mando == 'casa':
                    url = f"{API_URL}fixtures?team={time_id}&season={temporada}&venue=home&status=FT"
                else:
                    url = f"{API_URL}fixtures?team={time_id}&season={temporada}&venue=away&status=FT"
                r = requests.get(url, headers=headers, timeout=10)
                r.raise_for_status()
                data = r.json()
                jogos = data.get('response', [])
                return jogos[-10:] if len(jogos) > 10 else jogos, None
            except Exception as e:
                return [], f"Erro ao buscar jogos: {e}"
        
        with st.spinner("Buscando jogos do mandante..."):
            st.session_state.jogos_a, erro_ja = buscar_jogos(time_a_id, temporada_a, 'casa')
        with st.spinner("Buscando jogos do visitante..."):
            st.session_state.jogos_b, erro_jb = buscar_jogos(time_b_id, temporada_b, 'fora')
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Mandante - Últimos 10 jogos em casa")
            if erro_ja:
                st.error(erro_ja)
            elif st.session_state.jogos_a:
                for j in st.session_state.jogos_a:
                    with st.expander(f"{j['fixture']['date'][:10]} - vs {j['teams']['away']['name']} ({j['goals']['home']}x{j['goals']['away']})"):
                        st.write(f"Liga: {j['league']['name']} ({j['league']['country']})")
                        try:
                            stats_url = f"{API_URL}fixtures/statistics?fixture={j['fixture']['id']}"
                            r_stats = requests.get(stats_url, headers=headers, timeout=10)
                            r_stats.raise_for_status()
                            stats_data = r_stats.json().get('response', [])
                        except Exception as e:
                            stats_data = []
                        if stats_data:
                            for team_stats in stats_data:
                                team_name = team_stats.get('team', {}).get('name', 'Desconhecido')
                                st.markdown(f"**{team_name}**")
                                for item in team_stats['statistics']:
                                    nome = item['type']
                                    valor = item['value'] if item['value'] is not None else '-'
                                    st.write(f"{nome}: {valor}")
                        else:
                            st.info("Sem estatísticas detalhadas disponíveis para este jogo.")
            else:
                st.info("Nenhum jogo encontrado.")
        
        with col2:
            st.subheader("Visitante - Últimos 10 jogos fora")
            if erro_jb:
                st.error(erro_jb)
            elif st.session_state.jogos_b:
                for j in st.session_state.jogos_b:
                    with st.expander(f"{j['fixture']['date'][:10]} - vs {j['teams']['home']['name']} ({j['goals']['home']}x{j['goals']['away']})"):
                        st.write(f"Liga: {j['league']['name']} ({j['league']['country']})")
                        try:
                            stats_url = f"{API_URL}fixtures/statistics?fixture={j['fixture']['id']}"
                            r_stats = requests.get(stats_url, headers=headers, timeout=10)
                            r_stats.raise_for_status()
                            stats_data = r_stats.json().get('response', [])
                        except Exception as e:
                            stats_data = []
                        if stats_data:
                            for team_stats in stats_data:
                                team_name = team_stats.get('team', {}).get('name', 'Desconhecido')
                                st.markdown(f"**{team_name}**")
                                for item in team_stats['statistics']:
                                    nome = item['type']
                                    valor = item['value'] if item['value'] is not None else '-'
                                    st.write(f"{nome}: {valor}")
                        else:
                            st.info("Sem estatísticas detalhadas disponíveis para este jogo.")
            else:
                st.info("Nenhum jogo encontrado.")
    else:
        st.info("Selecione e busque os times na aba anterior.")

with abas[2]:
    st.header("Médias (Simples e Ponderadas)")
    def calcular_medias(jogos, tipo_mando):
        if not jogos:
            return pd.DataFrame(), pd.DataFrame()
        estatisticas = {}
        estatisticas_pond = {}
        for j in jogos:
            try:
                stats_url = f"{API_URL}fixtures/statistics?fixture={j['fixture']['id']}"
                r_stats = requests.get(stats_url, headers=headers, timeout=10)
                r_stats.raise_for_status()
                stats_data = r_stats.json().get('response', [])
            except Exception as e:
                stats_data = []
            if tipo_mando == 'casa':
                adversario = j['teams']['away']['name']
                campeonato = j['league']['name']
            else:
                adversario = j['teams']['home']['name']
                campeonato = j['league']['name']
            peso = PESOS.get(campeonato, PESOS.get('Outro', 0.5))
            for team_stats in stats_data:
                team_name = team_stats.get('team', {}).get('name', 'Desconhecido')
                for item in team_stats['statistics']:
                    nome = item['type']
                    valor = item['value']
                    if valor is None or not isinstance(valor, (int, float)):
                        continue
                    estatisticas.setdefault(team_name, {}).setdefault(nome, []).append(valor)
                    if team_name == (j['teams']['home']['name'] if tipo_mando == 'casa' else j['teams']['away']['name']):
                        estatisticas_pond.setdefault(team_name, {}).setdefault(nome, []).append(valor * peso)
                    else:
                        estatisticas_pond.setdefault(team_name, {}).setdefault(nome, []).append(valor / peso if peso else valor)
        medias_simples = {}
        for time, stats in estatisticas.items():
            medias_simples[time] = {k: round(sum(v)/len(v),2) for k,v in stats.items() if v}
        medias_pond = {}
        for time, stats in estatisticas_pond.items():
            medias_pond[time] = {k: round(sum(v)/len(v),2) for k,v in stats.items() if v}
        return pd.DataFrame(medias_simples).T, pd.DataFrame(medias_pond).T
    
    jogos_a = st.session_state.get('jogos_a', [])
    jogos_b = st.session_state.get('jogos_b', [])
    
    st.subheader('Mandante')
    df_simples_a, df_pond_a = calcular_medias(jogos_a, 'casa')
    st.write('Médias Simples')
    st.dataframe(df_simples_a)
    st.write('Médias Ponderadas')
    st.dataframe(df_pond_a)
    
    st.subheader('Visitante')
    df_simples_b, df_pond_b = calcular_medias(jogos_b, 'fora')
    st.write('Médias Simples')
    st.dataframe(df_simples_b)
    st.write('Médias Ponderadas')
    st.dataframe(df_pond_b)

with abas[3]:
    st.header("Estatísticas Previstas")
    def prever_estatisticas(df_pond_a, df_pond_b):
        if df_pond_a.empty or df_pond_b.empty:
            return pd.DataFrame()
        time_a = df_pond_a.index[0]
        time_b = df_pond_b.index[0]
        stats = sorted(list(set(df_pond_a.columns).intersection(set(df_pond_b.columns))))
        previsao = {}
        for stat in stats:
            feita_a = df_pond_a.loc[time_a, stat] if stat in df_pond_a.columns else 0
            sofrida_b = df_pond_b.loc[time_b, stat] if stat in df_pond_b.columns else 0
            prev_a = ((feita_a + sofrida_b) * 1.2) / 2
            feita_b = df_pond_b.loc[time_b, stat] if stat in df_pond_b.columns else 0
            sofrida_a = df_pond_a.loc[time_a, stat] if stat in df_pond_a.columns else 0
            prev_b = ((feita_b + sofrida_a) / 1.2) / 2
            previsao[stat] = {
                'Mandante': round(prev_a,2),
                'Visitante': round(prev_b,2)
            }
        return pd.DataFrame(previsao).T
    
    if 'df_pond_a' in locals() and 'df_pond_b' in locals():
        df_prev = prever_estatisticas(df_pond_a, df_pond_b)
        if not df_prev.empty:
            st.write('Previsão das Estatísticas da Partida')
            st.dataframe(df_prev)
            st.write('Intervalo de Confiança (aprox. 85%)')
            def ic_85(media, std):
                return (round(media - 1.44*std,2), round(media + 1.44*std,2))
            ic_dict = {}
            for stat in df_prev.index:
                std_a = df_pond_a[stat].std() if stat in df_pond_a.columns else 0
                std_b = df_pond_b[stat].std() if stat in df_pond_b.columns else 0
                ic_dict[stat] = {
                    'Mandante': ic_85(df_prev.loc[stat, 'Mandante'], std_a),
                    'Visitante': ic_85(df_prev.loc[stat, 'Visitante'], std_b)
                }
            st.dataframe(pd.DataFrame(ic_dict).T)
        else:
            st.info('Calcule as médias ponderadas primeiro.')
    else:
        st.info('Calcule as médias ponderadas primeiro.')

with abas[4]:
    st.header("Placar Provável")
    def prever_placar Wade(df_prev):
        if df_prev is None or df_prev.empty:
            return None, None, None, None
        if 'Gols' in df_prev.index:
            exp_gols_a = df_prev.loc['Gols', 'Mandante']
            exp_gols_b = df_prev.loc['Gols', 'Visitante']
        elif 'Goals' in df_prev.index:
            exp_gols_a = df_prev.loc['Goals', 'Mandante']
            exp_gols_b = df_prev.loc['Goals', 'Visitante']
        else:
            return None, None, None, None
        max_gols = 6
        prob_matrix = np.zeros((max_gols+1, max_gols+1))
        for i in range(max_gols+1):
            for j in range(max_gols+1):
                prob_matrix[i,j] = poisson.pmf(i, exp_gols_a) * poisson.pmf(j, exp_gols_b)
        idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
        placar_provavel = f"{idx[0]} x {idx[1]}"
        prob_vitoria_a = np.sum(np.tril(prob_matrix, -1))
        prob_vitoria_b = np.sum(np.triu(prob_matrix, 1))
        prob_empate = np.trace(prob_matrix)
        probs = {
            'Vitória Mandante': round(prob_vitoria_a*100,1),
            'Empate': round(prob_empate*100,1),
            'Vitória Visitante': round(prob_vitoria_b*100,1)
        }
        def intervalo_poisson(mu):
            std = np.sqrt(mu)
            return (round(mu-1.44*std,2), round(mu+1.44*std,2))
        ic_a = intervalo_poisson(exp_gols_a)
        ic_b = intervalo_poisson(exp_gols_b)
        return placar_provavel, probs, ic_a, ic_b
    
    if 'df_prev' in locals() and df_prev is not None and not df_prev.empty:
        placar, probs, ic_a, ic_b = prever_placar(df_prev)
        if placar:
            st.write(f'**Placar mais provável:** {placar}')
            st.write('**Probabilidades:**')
            st.write(probs)
            st.write('**Intervalo de confiança (85%) para gols:**')
            st.write({
                'Mandante': ic_a,
                'Visitante': ic_b
            })
        else:
            st.info('Calcule a previsão de gols primeiro.')
    else:
        st.info('Calcule a previsão de gols primeiro.')

with abas[5]:
    st.header("Odds x Previsão")
    def buscar_odds(fixture_id):
        try:
            url = f"{API_URL}odds?fixture={fixture_id}"
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json().get('response', [])
            return data
        except Exception as e:
            return []
    
    def odds_para_prob(odd):
        try:
            return round(100/(float(odd)),2) if odd and float(odd)>0 else 0
        except:
            return 0
    
    fixture_id = None
    if 'jogos_a' in st.session_state and st.session_state.jogos_a:
        fixture_id = st.session_state.jogos_a[0]['fixture']['id']
    odds_data = buscar_odds(fixture_id) if fixture_id else []
    
    if odds_data:
        mercados = {}
        for league in odds_data:
            for bookmaker in league.get('bookmakers', []):
                for bet in bookmaker.get('bets', []):
                    mercados[bet['name']] = bet['values']
        st.subheader('Odds dos principais mercados')
        for mercado, valores in mercados.items():
            st.markdown(f'**{mercado}**')
            odds_table = {v['value']: v['odd'] for v in valores}
            st.write(odds_table)
        if 'Match Winner' in mercados or 'Resultado' in mercados:
            mercado_nome = 'Match Winner' if 'Match Winner' in mercados else 'Resultado'
            odds_1x2 = {v['value']: float(v['odd']) for v in mercados[mercado_nome]}
            probs_1x2 = {k: odds_para_prob(v) for k,v in odds_1x2.items()}
            st.write('Probabilidades implícitas das odds (1x2):')
            st.write(probs_1x2)
            if 'df_prev' in locals() and df_prev is not None and not df_prev.empty:
                if 'prob_vitoria_a' in locals() and 'prob_empate' in locals() and 'prob_vitoria_b' in locals():
                    st.write('Probabilidades do modelo:')
                    st.write({
                        'Vitória Mandante': prob_vitoria_a,
                        'Empate': prob_empate,
                        'Vitória Visitante': prob_vitoria_b
                    })
                st.write('**Oportunidades de Valor:**')
                for k in ['Home', 'Draw', 'Away', 'Mandante', 'Empate', 'Visitante']:
                    if k in probs_1x2:
                        odd_prob = probs_1x2[k]
                        if k in ['Home', 'Mandante']:
                            modelo_prob = prob_vitoria_a if 'prob_vitoria_a' in locals() else 0
                        elif k in ['Draw', 'Empate']:
                            modelo_prob = prob_empate if 'prob_empate' in locals() else 0
                        elif k in ['Away', 'Visitante']:
                            modelo_prob = prob_vitoria_b if 'prob_vitoria_b' in locals() else 0
                        else:
                            modelo_prob = 0
                        if modelo_prob > odd_prob:
                            st.success(f"Aposta de valor em {k}: Modelo={modelo_prob:.1f}% > Odd Implícita={odd_prob:.1f}%")
    else:
        st.info('Odds não encontradas para o confronto.')

with abas[6]:
    st.header("Exportar Resultado")
    st.info("Ao final, exporte os dados para .xlsx.")
    def exportar_para_excel():
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            if 'df_simples_a' in locals() and not df_simples_a.empty:
                df_simples_a.to_excel(writer, sheet_name='Medias_Simples_Mandante')
            if 'df_simples_b' in locals() and not df_simples_b.empty:
                df_simples_b.to_excel(writer, sheet_name='Medias_Simples_Visitante')
            if 'df_pond_a' in locals() and not df_pond_a.empty:
                df_pond_action_a.to_excel(writer, sheet_name='Medias_Ponderadas_Mandante')
            if 'df_pond_b' in locals() and not df_pond_b.empty:
                df_pond_b.to_excel(writer, sheet_name='Medias_Ponderadas_Visitante')
            if 'df_prev' in locals() and not df_prev.empty:
                df_prev.to_excel(writer, sheet_name='Estatisticas_Previstas')
            if 'ic_dict' in locals() and ic_dict:
                pd.DataFrame(ic_dict).T.to_excel(writer, sheet_name='IC_85')
            if 'placar' in locals() and placar:
                pd.DataFrame({'Placar Provável': [placar]}).to_excel(writer, sheet_name='Placar_Provavel')
            if 'probs' in locals() and probs:
                pd.DataFrame([probs]).to_excel(writer, sheet_name='Probabilidades_Placar')
            if 'mercados' in locals() and mercados:
                for mercado, valores in mercados.items():
                    pd.DataFrame(valores).to_excel(writer, sheet_name=f'Odds_{mercado[:20]}')
            writer.save()
        output.seek(0)
        return output
    
    if st.button('Exportar para Excel (.xlsx)'):
        excel_data = exportar_para_excel()
        st.download_button(
            label='Baixar arquivo Excel',
            data=excel_data,
            file_name=f'previsao_futebol_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )