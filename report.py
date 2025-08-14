import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 페이지 기본 설정 
st.set_page_config(layout="wide")

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

try:
    
    df = pd.read_csv('anodizing_data.csv', sep=',')
    df1 = pd.read_csv('failure_df.csv', sep=',')
    merged = df.merge(df1, on='sequence_index')
except FileNotFoundError:
    st.error("데이터 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    st.stop()

#----------------------------------------------------------------------------데이터 전처리-------------------------------------------------------------------------------
merged['pk_datetime'] = pd.to_datetime(merged['pk_datetime'])
merged['power_watts'] = merged['volt'] * merged['ampere']
merged['time_diff_seconds'] = merged.groupby('sequence_index')['pk_datetime'].diff().dt.total_seconds().fillna(0)
merged['heat_joules'] = merged['power_watts'] * merged['time_diff_seconds']

#----------------------------------------------------------------------------모델 학습 및 캐싱----------------------------------------------------------------------------
@st.cache_data
def train_model():
    """
    모델을 학습하고 테스트 정확도를 계산합니다.
    """
    summary_df = merged.groupby('sequence_index').agg(
        volt=('volt', 'mean'),
        ampere=('ampere', 'mean'),
        temperature=('temperature', 'mean'),
        failure=('failure', 'first')
    ).reset_index()

    X = summary_df.drop(columns=['failure', 'sequence_index'])
    y = summary_df['failure']
    y = y.map({-1: 0, 1: 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

#-------------------------------------------------------------------------테스트 데이터셋으로 정확도 계산--------------------------------------------------------------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

#----------------------------------------------------------------수정된 함수를 호출하여 모델과 정확도 값을 동시에 받습니다.------------------------------------------------------------
model, model_accuracy = train_model()

#----------------------------------------------------------------------------KPI 대시보드 데이터 처리-----------------------------------------------------------------------------
merged['Date'] = merged.groupby('sequence_index')['pk_datetime'].transform('min').dt.date
seq_df = merged[['Date', 'sequence_index', 'failure', 'rec_num']].drop_duplicates(subset=['Date', 'sequence_index'])
kpi_df = seq_df.groupby('Date').agg(
    total_sequences=('sequence_index', 'nunique'),
    normal_sequences=('failure', lambda x: (x == 1).sum()),
    failed_sequences=('failure', lambda x: (x == -1).sum())
).reset_index()
kpi_df['failure_rate'] = (kpi_df['failed_sequences'] / kpi_df['total_sequences']) * 100
date_range = pd.date_range(start=kpi_df['Date'].min(), end=kpi_df['Date'].max())
kpi_df = kpi_df.set_index('Date').reindex(date_range, fill_value=0).rename_axis('Date').reset_index()


#----------------------------------------------------------------------------전체 기간 KPI 계산------------------------------------------------------------------------------
total_seq_all_time = kpi_df['total_sequences'].sum()
total_normal_all_time = kpi_df['normal_sequences'].sum()
total_failed_all_time = kpi_df['failed_sequences'].sum()
failure_rate_all_time = (total_failed_all_time / total_seq_all_time) * 100 if total_seq_all_time > 0 else 0

#--------------------------------------------------------------공정 효율성 지표: 시퀀스당 평균 공정 시간 계산--------------------------------------------------------------------
sequence_duration = merged.groupby('sequence_index').agg(
    start_time=('pk_datetime', 'min'),
    end_time=('pk_datetime', 'max'),
    failure=('failure', 'first')
).reset_index()
sequence_duration['duration_seconds'] = (sequence_duration['end_time'] - sequence_duration['start_time']).dt.total_seconds()
avg_duration_normal = sequence_duration[sequence_duration['failure'] == 1]['duration_seconds'].mean()
avg_duration_failed = sequence_duration[sequence_duration['failure'] == -1]['duration_seconds'].mean()
delta_duration = avg_duration_normal-avg_duration_failed
#--------------------------------------------------------------------------대시보드 헤더-------------------------------------------------------------------------
st.header("2² 양극산화피막 공정 현황 및 불량 예측")









st.markdown("---")
#---------------------------------------------------------------레이아웃 수정: 최상단 KPI 지표 일렬 배치-------------------------------------------------------------------------
st.header("KPI 지표")
kpi_cols = st.columns(6)

with kpi_cols[0]:
    with st.container(border=True):
        st.metric("총 시퀀스 수", f"{total_seq_all_time:,}")

with kpi_cols[1]:
    with st.container(border=True):
        st.metric("정상 시퀀스 수", f"{total_normal_all_time:,}")

with kpi_cols[2]:
    with st.container(border=True):
        st.metric("불량 시퀀스 수", f"{total_failed_all_time:,}")

with kpi_cols[3]:
    with st.container(border=True):
        st.metric("전체 불량률", f"{failure_rate_all_time:.2f}%",delta=f"{failure_rate_all_time:.2f}%", delta_color="inverse")

with kpi_cols[4]:
    with st.container(border=True):
        st.metric("정상 평균 공정 시간", f"{avg_duration_normal:.0f} s" if not pd.isna(avg_duration_normal) else "N/A",delta=f"{delta_duration:.2f}",delta_color="normal")

with kpi_cols[5]:
    with st.container(border=True):
        st.metric("불량 평균 공정 시간", f"{avg_duration_failed:.0f} s" if not pd.isna(avg_duration_failed) else "N/A",f"{-abs(delta_duration):.2f}",delta_color="normal")

st.markdown("---")

#----------------------------------------------------------------------------2행 레이아웃 조정------------------------------------------------------------------------------
pie_chart_cols = st.columns(3)
pie_chart_cols = st.columns([2, 1, 1])
#--------------------------------------------------------------------1번 컬럼: 날짜별 시퀀스 현황 바 차트---------------------------------------------------------------------
with pie_chart_cols[0]:
    st.subheader("날짜별 시퀀스 현황")
    with st.container(border=True):
        st.bar_chart(kpi_df.set_index('Date')[['normal_sequences', 'failed_sequences']], height=388, use_container_width=True)

#------------------------------------------------------------------2번 컬럼: 전체 불량 시퀀스 비율 파이 차트-------------------------------------------------------------------
with pie_chart_cols[1]:
    st.subheader("전체 불량 시퀀스 비율")
    with st.container(border=True):
        fig, ax = plt.subplots(figsize=(6, 8)) 
        
        # 배경 투명 처리
        fig.patch.set_alpha(0.0)   # 전체 figure 배경
        ax.set_facecolor("none")   # 차트 내부 배경
        
        sizes = [total_normal_all_time, total_failed_all_time]
        labels = ['정상', '불량']
        colors = ['#1f77b4', '#aec7e8']  
        
        wedges, texts, autotexts = ax.pie(
            sizes, colors=colors, autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(width=0.4, edgecolor='w'),
            textprops={'fontsize': 14, 'fontweight': 'bold'}  # 글자 크기 & 볼드
        )
        # 파이차트 수치 위치 조정
        for autotext in autotexts:
            x, y = autotext.get_position()
            autotext.set_position((x * 1.3, y * 1.3))
        
        # 범례 스타일
        ax.legend(labels, loc="center", title="Status", bbox_to_anchor=(0.5, 0.5),
                  prop={'size': 12, 'weight': 'bold'}, title_fontsize=14)
        
        st.pyplot(fig)

# ----------------------------------------------------------------
with pie_chart_cols[2]:
    st.subheader("정류기별 불량 발생률")
    with st.container(border=True):
        rec_failure_counts = seq_df[seq_df['failure'] == -1]['rec_num'].value_counts()
        
        if not rec_failure_counts.empty:
            fig, ax = plt.subplots(figsize=(6, 8)) 
            
            # 배경 투명 처리
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")
            
            labels = rec_failure_counts.index.astype(str).tolist()
            colors = ['#97B6D9', '#C2C9D1'] 
            
            wedges, texts, autotexts = ax.pie(
                rec_failure_counts.values, colors=colors, autopct='%1.1f%%', startangle=90,
                wedgeprops=dict(width=0.4, edgecolor='w'),
                textprops={'fontsize': 14, 'fontweight': 'bold'}
            )
            
            # 파이차트 수치 위치 조정
            for autotext in autotexts:
                x, y = autotext.get_position()
                autotext.set_position((x * 1.3, y * 1.3))
            
            ax.legend(labels, loc="center", title="Rec. Num", bbox_to_anchor=(0.5, 0.5),
                      prop={'size': 12, 'weight': 'bold'}, title_fontsize=14)
            
            st.pyplot(fig)
        else:
            st.info("불량 시퀀스가 없어 REC별 불량률을 표시할 수 없습니다.")


st.markdown("---")

#----------------------------------------------------------레이아웃 조정: 2개 열에 불량 시퀀스 상세 정보와 표 배치-------------------------------------------------------------
bottom_cols = st.columns(2)

#----------------------------------------------------------------------불량 시퀀스 상세 정보-------------------------------------------------------------------------------
with bottom_cols[0]:
    st.subheader("불량 시퀀스 상세 정보")
    with st.container(border=True):
        st.write("특정 날짜를 선택하여 해당 날짜의 불량 시퀀스 Index를 확인하세요.")
        if not merged.empty:
            min_date = merged['pk_datetime'].min().date()
            selected_date = st.date_input("날짜를 선택하세요", value=min_date, key='date_select')
            
            # 선택된 날짜의 불량 시퀀스 데이터 필터링
            defective_sequences_by_date = merged[(merged['pk_datetime'].dt.date == selected_date) & (merged['failure'] == -1)].copy()
            defective_sequences_by_date = defective_sequences_by_date[['sequence_index', 'rec_num']].drop_duplicates().reset_index(drop=True)
            
            if not defective_sequences_by_date.empty:
                # 1. REC_num 컬럼을 추가하여 데이터프레임 생성
                result_df = defective_sequences_by_date.rename(columns={'sequence_index': '불량 Sequence Index', 'rec_num': 'REC 번호'})
                st.dataframe(result_df)
                
                # 2. 불량 발생 건수 문구 추가
                rec_counts = defective_sequences_by_date['rec_num'].value_counts().sort_index()
                total_failed_count = rec_counts.sum()
                rec_counts_str = ", ".join([f"rec_no.{rec} : {count}건" for rec, count in rec_counts.items()])
                
                #  st.warning
                st.warning(f"⚠️ 총 **{total_failed_count}건**의 불량이 발생하였습니다. (세부 내역: **{rec_counts_str}**)")
                
            else:
                st.info(f"{selected_date}에 불량으로 판정된 시퀀스가 없습니다.")
        else:
            st.info("데이터가 비어있습니다. 데이터를 먼저 로드해주세요.")

#----------------------------------------------------------------------------날짜별 생산 현황 표---------------------------------------------------------------------------
with bottom_cols[1]:
    st.subheader("날짜별 생산 현황")
    with st.container(border=True):
        st.dataframe(kpi_df.set_index('Date'), use_container_width=True, height=305)

st.markdown("---")


#-----------------------------------------------------------------------시계열 데이터와 결함 예측---------------------------------------------------------------------------
bottom_left_col, bottom_right_col = st.columns(2)

#----------------------------------------------------------------------------시계열 데이터 분석----------------------------------------------------------------------------
with bottom_left_col:
    st.header("시계열성 데이터 분석")
    
    # 데이터 그룹핑 
    normal_sequences = sorted(merged[merged['failure'] == 1]['sequence_index'].unique())
    failed_sequences = sorted(merged[merged['failure'] == -1]['sequence_index'].unique())

    def group_sequences(seq_list, group_size=11):
        return {
            f'Group {i+1}': seq_list[i*group_size:(i+1)*group_size]
            for i in range(len(seq_list) // group_size + (1 if len(seq_list) % group_size > 0 else 0))
        }

    normal_groups = group_sequences(normal_sequences)
    failed_groups = group_sequences(failed_sequences)

    # 컨트롤러(필터) 구역 
    with st.container(border=True):
        col_metric, col_status, col_group, col_seq_select = st.columns([2,2,2,2])
        with col_metric:
            selected_metric = ''
            with st.popover("컬럼 선택"):
                for metric in ('volt', 'ampere', 'temperature', 'heat_joules'):
                    if st.checkbox(metric, value=(metric=='volt'), key=f"check_metric_{metric}"):
                        selected_metric = metric
        with col_status:
            status_choice = ''
            with st.popover("시퀀스 선택"):
                if st.checkbox("정상", value=True, key="check_status_정상"):
                    status_choice = '정상'
                if st.checkbox("불량", value=False, key="check_status_불량"):
                    status_choice = '불량'
        with col_group:
            group_choice = ''
            if status_choice == '정상':
                sequence_groups = normal_groups
            else:
                sequence_groups = failed_groups
            with st.popover("그룹 선택"):
                for group in list(sequence_groups.keys()):
                    if st.checkbox(group, value=(group==list(sequence_groups.keys())[0]), key=f"check_group_{group}"):
                        group_choice = group
        with col_seq_select:
            selected_sequences_in_group = sequence_groups.get(group_choice, [])
            selected_sequence_ids = []
            with st.popover("개별 선택"):
               
                for seq_id in selected_sequences_in_group:
                    if st.checkbox(f"Seq {seq_id}", value=True, key=f"check_{seq_id}"):
                        selected_sequence_ids.append(seq_id)
    # 시각화 
    with st.container(border=True):
        st.subheader(f"{status_choice} 시퀀스 {group_choice} {selected_metric} 변화 (공정 중반까지)")
        if selected_sequence_ids and selected_metric:
            sequences_to_plot = selected_sequence_ids
            fig, ax = plt.subplots(figsize=(12, 6))

            for seq_id in sequences_to_plot:
                seq_data = merged[merged['sequence_index'] == seq_id].copy()
                if not seq_data.empty:
                    seq_data['elapsed_time'] = (seq_data['pk_datetime'] - seq_data['pk_datetime'].min()).dt.total_seconds()
                    half_duration = seq_data['elapsed_time'].max() / 2
                    half_seq = seq_data[seq_data['elapsed_time'] <= half_duration]
                    ax.plot(half_seq['elapsed_time'], half_seq[selected_metric], label=f'Seq {seq_id}')

            ax.set_xlabel('경과 시간 (sec)')
            ax.set_ylabel(f'{selected_metric} 변화')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.info("시각화할 시퀀스를 하나 이상 선택해주세요.")

#----------------------------------------------------------------------------결함 예측 코드----------------------------------------------------------------------------
with bottom_right_col:
    st.header("결함 예측")
    #  정확도 컨테이너 추가
    with st.container(border=True):
        
        # st.markdown과 HTML/CSS를 사용하여 텍스트 크기와 간격을 세밀하게 조정합니다.
        st.markdown(f"""
            <style>
            .custom-metric-container {{
                line-height: 1.2; /* 전체적인 줄 간격을 줄여 높이를 조절합니다. */
            }}
            .custom-metric-container .title {{
                font-size: 0.9rem; /* '모델 예측 정확도' 글자 크기 */
                color: #fafafa; /* Streamlit 기본 텍스트 색상 */
                margin-bottom: 0.1rem; /* 제목과 값 사이 간격 줄이기 */
            }}
            .custom-metric-container .value-row {{
                display: flex;
                align-items: baseline;
                gap: 8px;
            }}
            .custom-metric-container .metric-value {{
                font-size: 2.0rem; /* 메트릭 값 글자 크기 (기존 2.25rem) */
                font-weight: 600;
            }}
            .custom-metric-container .metric-caption {{
                font-size: 0.8rem; /* 캡션 글자 크기 (기존 0.875rem) */
                color: #808495;
            }}
            </style>
            <div class="custom-metric-container">
                <div class="title">모델 예측 정확도</div>
                <div class="value-row">
                    <span class="metric-value">{model_accuracy*100:.2f}%</span>
                    <span class="metric-caption">테스트 데이터셋 기준</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.write("아래 슬라이더를 조정하여 공정 조건을 입력하고, 결함 여부를 예측해 보세요.")

    input_volt = st.slider("전압 (V)", 0.0, 31.1, 15.0, 0.1)
    input_ampere = st.slider("전류 (A)", 0, 1522, 750, 1)
    input_temp = st.slider("온도 (℃)", 9, 18, 12, 1)

    # 평균값 계산
    avg_volt = merged['volt'].mean()
    avg_ampere = merged['ampere'].mean()
    avg_temp = merged['temperature'].mean()

    if st.button("예측하기"):
        input_data = pd.DataFrame([[input_volt, input_ampere, input_temp]],
                                   columns=['volt', 'ampere', 'temperature'])
        
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        # 불량 예측 시
        if prediction[0] == 0:
            st.warning(f"예측 결과: 불량 (확률: {prediction_proba[0][0]*100:.2f}%)")
            
            high_values = []
            if input_volt > avg_volt:
                high_values.append("전압")
            if input_ampere > avg_ampere:
                high_values.append("전류")
            if input_temp > avg_temp:
                high_values.append("온도")
            
            if high_values:
                st.error(f"⛔️ 불량 예측의 원인으로 분석된 공정 조건은 다음과 같습니다: {', '.join(high_values)} 값이 평균보다 높습니다.")
            else:
                st.info("⛔️ 불량으로 예측되었지만, 입력된 공정 조건이 평균보다 높지 않습니다. 다른 요인을 확인해 보세요.")

        # 정상 예측 시
        else:
            st.success(f"예측 결과: 정상 (확률: {prediction_proba[0][1]*100:.2f}%)")

            # 평균보다 높은 값이 있는지 확인
            high_values = []
            if input_volt > avg_volt:
                high_values.append("전압")
            if input_ampere > avg_ampere:
                high_values.append("전류")
            if input_temp > avg_temp:
                high_values.append("온도")

            if high_values:
                st.warning(f"⚠️ 현재 공정은 정상이나, {', '.join(high_values)} 값이 평균보다 높아 주의가 필요합니다.")
            else:
                st.success("✅️ 현재 공정은 적정 수준에서 진행되고 있습니다.")

st.markdown("---")
