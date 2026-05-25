from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login ,logout,authenticate
from django.core.files.storage import default_storage
import matplotlib.pyplot as plt
import pandas as pd
import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, BatchNormalization
from xgboost import XGBClassifier
import joblib
from tensorflow.keras.layers import GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, Add, BatchNormalization, Activation, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Required for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Declare label list (used in your classification report & confusion matrix)
Label = ["failure","normal"]

# Create your views here.
def home(request):
    return render(request,'Home.html')

from django.shortcuts import render

def about(request):
    return render(request, 'about.html')

def register(request):
    if request.method == 'POST':
        First_Name = request.POST['name']
        Email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']
        confirmation_password = request.POST['cnfm_password']
        select_user=request.POST['role']
        if select_user=='admin':
            admin=True
        else:
            admin=False
        if password == confirmation_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists, please choose a different one.')
                return redirect('register')
            else:
                if User.objects.filter(email=Email).exists():
                    messages.error(request, 'Email already exists, please choose a different one.')
                    return redirect('register')
                else:
                    user = User.objects.create_user(
                        username=username,
                        password=password,
                        email=Email,
                        first_name=First_Name,
                        is_staff=admin
                    )
                    user.save()
                    return redirect('login')
        else:
            messages.error(request, 'Passwords do not match.')
        return render(request, 'register.html')
    return render(request, 'register.html')
def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        if User.objects.filter(username=username).exists():
            user=User.objects.get(username=username)
            if user.check_password(password):
                user = authenticate(username=username,password=password)
                if user is not None:
                    login(request,user)
                    messages.success(request,'login successfull')
                    return redirect('/')
                else:
                   messages.error(request,'please check the Password Properly')
                   return redirect('login')
            else:
                messages.error(request,"please check the Password Properly")  
                return redirect('login') 
        else:
            messages.error(request,"username doesn't exist")
            return redirect('login')
    return render(request,'login.html')
def logout_view(request):
    logout(request)
    return redirect('login')

import os

# Ensure 'static/images' directory exists
os.makedirs(os.path.join('static', 'images'), exist_ok=True)
chart_path = os.path.join('static', 'images', 'performance_chart.png')


from django.shortcuts import render
from django.core.files.storage import default_storage
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Global variables
X = y = df = None
le = LabelEncoder()
labels = ["failure","normal"]  # Based on earlier context

# ----------------- UPLOAD FUNCTION -----------------
def Upload_data(request):
    load = True
    global df, x, y

    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)

        # Read uploaded file
        df = pd.read_csv(default_storage.path(file_path), low_memory=False)
        df.drop(["pump_status"], axis=1, inplace=True)


        # Drop unwanted columns
        

        # Delete file after reading
        default_storage.delete(file_path)

        # Preview uploaded data
        outdata = df.head(100)

        return render(request, 'prediction.html', {'predict': outdata.to_html(classes='table table-striped')})

    return render(request, 'prediction.html', {'upload': load})

# ----------------- PREPROCESS FUNCTION -----------------
# ----------------- PREPROCESS FUNCTION -----------------
global x_train, x_test, y_train, y_test
def preprocess(request):
    global df, x, y, le,x_train,x_test,y_train,y_test

    # Label encode categorical columns
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            df[col] = le.fit_transform(df[col].astype(str))

    # Save processed DataFrame
    df_encoded =df.copy()
    df_encoded['maintenance_required'] = y

    # Save to CSV
    df_encoded.to_csv('agri_predictive_maintenance_encoded.csv', index=False)

    # Preview first 10 rows
    preview_df = df_encoded.head(10)
    x= df.drop('maintenance_required', axis=1)
    y = df['maintenance_required']
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

    return render(request, 'prediction.html', {
        'message': '✅ Preprocessing completed and credit_card_fraud_encoded.csv saved.',
        'preview': preview_df.to_html(classes='table table-bordered table-sm')
    })


from django.shortcuts import render
import numpy as np
import pandas as pd
import random
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Global variables to access processed features
#X_selected = y_selected = None







precision = []
recall = []
fscore = []
accuracy = []

def calculateMetrics(algorithm, predict, testY):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix

    # Ensure testY and predict are integers
    testY = testY.astype('int64')
    predict = predict.astype('int64')

    # Calculate metrics
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    # Print metrics
    print(f'{algorithm} Accuracy    : {a}')
    print(f'{algorithm} Precision   : {p}')
    print(f'{algorithm} Recall      : {r}')
    print(f'{algorithm} FSCORE      : {f}')

    # Classification report
    report = classification_report(testY, predict, target_names=Label)
    print(f'\n{algorithm} classification report\n{report}')

    # Confusion matrix
    conf_matrix = confusion_matrix(testY, predict)

    # Save confusion matrix plot to static/images/
    os.makedirs('static/images', exist_ok=True)
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(conf_matrix, xticklabels=Label, yticklabels=Label, annot=True, cmap="Spectral", fmt="g")
    ax.set_ylim([0, len(Label)])
    plt.title(f'{algorithm} Confusion Matrix')
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.savefig(f'static/images/{algorithm}_confusion_matrix.png')  # Save image
    plt.close()  # Close figure to release memory

    #threading.Thread(target=show_plot).start()

        
        
def train_lstm_view(request):
    import os
    import numpy as np
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping

    # Ensure the model directory exists
    os.makedirs('model', exist_ok=True)
    model_path = 'model/LSTM_Model.h5'

    # Reshape x_train and x_test for LSTM
    x_train_lstm = x_train.values.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test_lstm = x_test.values.reshape((x_test.shape[0], 1, x_test.shape[1]))

    # Load or train the model
    if os.path.exists(model_path):
        lstm_model = load_model(model_path)
        print("✅ LSTM model loaded successfully.")
    else:
        print("❌ Model not found. Training LSTM model...")
        timesteps = x_train_lstm.shape[1]
        features = x_train_lstm.shape[2]

        lstm_model = Sequential()
        lstm_model.add(LSTM(64, input_shape=(timesteps, features), return_sequences=False))
        lstm_model.add(Dense(32, activation='relu'))
        lstm_model.add(Dense(1, activation='sigmoid'))  # For binary classification

        lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        lstm_model.fit(
            x_train_lstm, y_train,
            epochs=50, batch_size=64,
            validation_split=0.2,
            callbacks=[early_stop]
        )

        lstm_model.save(model_path)
        print("✅ LSTM model trained and saved.")

    # Prediction
    y_pred = (lstm_model.predict(x_test_lstm) > 0.5).astype(int)

    # Evaluation using shared metrics function
    calculateMetrics("LSTM_Model", y_pred, y_test)

    return render(request, 'prediction.html', {
        'algorithm': 'LSTM_Model',
        'accuracy': f"{accuracy[-1]:.2f}%",
        'precision': f"{precision[-1]:.2f}%",
        'recall': f"{recall[-1]:.2f}%",
        'fscore': f"{fscore[-1]:.2f}%"
    })



def train_bilstm_view(request):
    import os
    import numpy as np
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping

    # Ensure the model directory exists
    os.makedirs('model', exist_ok=True)
    model_path = 'model/BiLSTM_Model.h5'

    # Reshape x_train and x_test for BiLSTM
    x_train_bilstm = x_train.values.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test_bilstm = x_test.values.reshape((x_test.shape[0], 1, x_test.shape[1]))

    # Load or train the model
    if os.path.exists(model_path):
        bilstm_model = load_model(model_path)
        print("✅ BiLSTM model loaded successfully.")
    else:
        print("❌ Model not found. Training BiLSTM model...")
        timesteps = x_train_bilstm.shape[1]
        features = x_train_bilstm.shape[2]

        bilstm_model = Sequential()
        bilstm_model.add(Bidirectional(LSTM(64, return_sequences=False), input_shape=(timesteps, features)))
        bilstm_model.add(Dense(32, activation='relu'))
        bilstm_model.add(Dense(1, activation='sigmoid'))  # For binary classification

        bilstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        bilstm_model.fit(
            x_train_bilstm, y_train,
            epochs=50, batch_size=64,
            validation_split=0.2,
            callbacks=[early_stop]
        )

        bilstm_model.save(model_path)
        print("✅ BiLSTM model trained and saved.")

    # Prediction
    y_pred = (bilstm_model.predict(x_test_bilstm) > 0.5).astype(int)

    # Evaluation using shared metrics function
    calculateMetrics("BiLSTM_Model", y_pred, y_test)

    return render(request, 'prediction.html', {
        'algorithm': 'BiLSTM_Model',
        'accuracy': f"{accuracy[-1]:.2f}%",
        'precision': f"{precision[-1]:.2f}%",
        'recall': f"{recall[-1]:.2f}%",
        'fscore': f"{fscore[-1]:.2f}%"
    })


def train_ann_view(request):
    import os
    import numpy as np
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.callbacks import EarlyStopping

    # Ensure the model directory exists
    os.makedirs('model', exist_ok=True)
    model_path = 'model/ANN_Model.h5'

    # ANN uses original features (no reshaping needed)
    x_train_ann = x_train.values
    x_test_ann = x_test.values

    # Load or train the model
    if os.path.exists(model_path):
        ann_model = load_model(model_path)
        print("✅ ANN model loaded successfully.")
    else:
        print("❌ Model not found. Training ANN model...")
        features = x_train_ann.shape[1]

        ann_model = Sequential()
        ann_model.add(Dense(128, activation='relu', input_dim=features))
        ann_model.add(Dense(64, activation='relu'))
        ann_model.add(Dense(32, activation='relu'))
        ann_model.add(Dense(1, activation='sigmoid'))  # For binary classification

        ann_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        ann_model.fit(
            x_train_ann, y_train,
            epochs=50, batch_size=64,
            validation_split=0.2,
            callbacks=[early_stop]
        )

        ann_model.save(model_path)
        print("✅ ANN model trained and saved.")

    # Prediction
    y_pred = (ann_model.predict(x_test_ann) > 0.5).astype(int)

    # Evaluation using shared metrics function
    calculateMetrics("ANN_Model", y_pred, y_test)

    return render(request, 'prediction.html', {
        'algorithm': 'ANN_Model',
        'accuracy': f"{accuracy[-1]:.2f}%",
        'precision': f"{precision[-1]:.2f}%",
        'recall': f"{recall[-1]:.2f}%",
        'fscore': f"{fscore[-1]:.2f}%"
    })


def train_lgbm_view(request):
    import os
    import numpy as np
    import lightgbm as lgb
    from joblib import dump, load

    # Ensure the model directory exists
    os.makedirs('model', exist_ok=True)
    model_path = 'model/LGBM_Model.pkl'

    # Convert x_train and x_test to numpy arrays
    x_train_lgb = x_train.values
    x_test_lgb = x_test.values
    y_train_lgb = y_train.values
    y_test_lgb = y_test.values

    # Load or train the model
    if os.path.exists(model_path):
        lgbm_model = load(model_path)
        print("✅ LGBM model loaded successfully.")
    else:
        print("❌ Model not found. Training LGBM model...")

        lgbm_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            random_state=42
        )

        # Train the model
        lgbm_model.fit(
            x_train_lgb, y_train_lgb,
            eval_set=[(x_test_lgb, y_test_lgb)],
            eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        # Save the model
        dump(lgbm_model, model_path)
        print("✅ LGBM model trained and saved.")

    # Prediction
    y_pred = lgbm_model.predict(x_test_lgb)

    # Evaluation using shared metrics function
    calculateMetrics("LGBM_Model", y_test_lgb, y_pred)

    return render(request, 'prediction.html', {
        'algorithm': 'LGBM_Model',
        'accuracy': f"{accuracy[-1]:.2f}%",
        'precision': f"{precision[-1]:.2f}%",
        'recall': f"{recall[-1]:.2f}%",
        'fscore': f"{fscore[-1]:.2f}%"
    })


import os
import pandas as pd
import matplotlib.pyplot as plt
from django.shortcuts import render

# Global metrics (you should ensure these are defined somewhere before this view is called)
accuracy = []
precision = []
recall = []
fscore = []

def performance_summary_view(request):
    global accuracy, precision, recall, fscore

    algorithm_names = ["LSTM", "BI_LSTM", "ANN", "LGBM"]

    if len(accuracy) < len(algorithm_names):
        return render(request, 'prediction.html', {'error': '❌ Not all models have been evaluated yet.'})

    columns = ["Algorithm Name", "Precision", "Recall", "F1 Score", "Accuracy"]
    values = []

    for i in range(len(algorithm_names)):
        values.append([
            algorithm_names[i],
            round(precision[i], 2),
            round(recall[i], 2),
            round(fscore[i], 2),
            round(accuracy[i], 2)
        ])

    temp = pd.DataFrame(values, columns=columns)

    # Save performance chart to static/images/
    static_images_dir = os.path.join('static', 'images')
    os.makedirs(static_images_dir, exist_ok=True)  # Ensure directory exists

    chart_path = os.path.join(static_images_dir, 'performance_chart.png')

    # Plot the chart
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Performance Metrics of Different Models", fontsize=16)

    axs[0, 0].bar(temp["Algorithm Name"], temp["Precision"], color='skyblue')
    axs[0, 0].set_title("Precision")
    axs[0, 0].tick_params(axis='x', rotation=15)

    axs[0, 1].bar(temp["Algorithm Name"], temp["Recall"], color='orange')
    axs[0, 1].set_title("Recall")
    axs[0, 1].tick_params(axis='x', rotation=15)

    axs[1, 0].bar(temp["Algorithm Name"], temp["F1 Score"], color='green')
    axs[1, 0].set_title("F1 Score")
    axs[1, 0].tick_params(axis='x', rotation=15)

    axs[1, 1].bar(temp["Algorithm Name"], temp["Accuracy"], color='purple')
    axs[1, 1].set_title("Accuracy")
    axs[1, 1].tick_params(axis='x', rotation=15)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(chart_path)
    plt.close()

    return render(request, 'prediction.html', {
        'table': temp.to_html(classes='table table-bordered table-hover', index=False),
        'chart_url': '/static/images/performance_chart.png'
    })

def predict_view(request):
    if request.method == 'POST' and request.FILES.get('file'):
        import pandas as pd
        import numpy as np
        import joblib
        from django.core.files.storage import default_storage

        labels = ["failure","normal"]

        # Read uploaded CSV file
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        test = pd.read_csv(default_storage.path(file_path))
        default_storage.delete(file_path)

        # Ensure 16 features
        if test.shape[1] < 8:
            for i in range(8- test.shape[1]):
                test[f'dummy_{i}'] = 0
        elif test.shape[1] > 8:
            test = test.iloc[:, :8]

        # Convert to array (2D)
        test_array = test.values

        # Load stacked model
        model = joblib.load('model/LGBM_Model.pkl')

        # Predict
        predictions = model.predict(test_array)

        # Map predictions to labels
        test['Predicted_Label'] = [labels[int(p)] for p in predictions]

        return render(request, 'prediction.html', {
            'predict': test.head(100).to_html(classes='table table-bordered', index=False)
        })

    return render(request, 'prediction.html', {'test': True})

