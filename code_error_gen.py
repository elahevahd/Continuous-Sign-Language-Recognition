from operator import itemgetter
from collections import Counter
import csv
import datetime
import itertools
import os
from operator import itemgetter
from collections import Counter
# For Dropbox API
import sys
import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError
import csv
import datetime
import itertools



message_dict={
'error_whq_lexical': 'When asking a WH question, dont forget to lean forward (a little) and lower your eyebrows.',
'error_ynq_lexical': 'When asking a yes/no question, dont forget to lean forward (a little) and raise your eyebrows.',
'error_neg_lexical':	'When indicating something negative, dont forget to furrow your eyebrows and shake your head (a little).',
'error_cond_lexical':	'When starting with a time, conditional (if/when), or topic, raise your eyebrows at the beginning of the sentence.',
'error_ynq_beginning':	'When asking a yes/no :question, dont forget to lean forward and raise your eyebrows during the sentence.',
'error_ynq_end':	'At the end of a yes/no question, you should be leaning forward (a little) with your eyebrows up.',
'error_cond_beginning':	'When starting with a time, conditional (if/when), or topic, raise your eyebrows at the beginning of the sentence.',
'error_topic_beginning': 'When starting with a time, conditional (if/when), or topic, raise your eyebrows at the beginning of the sentence.',
'error_ynq_sign_during_whq': 'The QUESTION or QUESTION-wiggle sign is optional at the end of Yes/No question, but it is not used at the end of WH questions.  Remember that Yes/No questions used an eyebrows-up face, and WH questions use an eyebrows-down face.',
'correct_topic_or_cond': 'It looks like you correctly raised your eyebrows at the beginning of a sentence when expressing a time, conditional (if/when), or topic.',
'correct_neg':	'It looks like you discussed something negative while using appropriate face and head movements, i.e. furrowing your eyebrows and shaking your head (a little).',
'correct_whq':	'It looks like you used a wh question facial expression correctly, i.e. leaning forward (a little) and lowering your eyebrows.',
'correct_ynq':	'It looks like you used a yes/no question facial expression correctly, i.e. leaning forward (a little) and raising your eyebrows.  Note: It is not always necessary to end such questions with the QUESTION sign.'
}


clause_confidence= 0
clause_confidence_lexical = 80

clause_distance_lexical = 10 
clause_distance_begin = 50
clause_distance_end = 50

face_confidence_lexical=0
head_confidence_lexical=0 
lexical_operator='OR'

face_confidence_begin_end=80
head_confidence_begin_end=80

gesture_confidence = 80


lexical_error_dict = {
'QUESTION_WH': [['QUESTION_WH','RHQ'],['YN-WH','COND-RHQ-Topic'], 'error_whq_lexical'],
'QUESTION_YN': [['QUESTION_YN','QUESTION_WH'],['YN-WH'], 'error_ynq_lexical'],
'NEGATIVE': [['NEGATIVE'],['NEGATIVE'],'error_neg_lexical'],
'CONDITIONAL': [['CONDITIONAL','TOPIC'],['COND-RHQ-Topic'],'error_cond_lexical'],
'TIME': [['CONDITIONAL','TOPIC'],['COND-RHQ-Topic'],'error_cond_lexical']
}

lexical_correct_dict = {
'QUESTION_WH': [['QUESTION_WH'],['YN-WH'], 'correct_whq'],
'QUESTION_YN': [['QUESTION_YN'],['YN-WH'], 'correct_ynq'],
'NEGATIVE': [['NEGATIVE'],['NEGATIVE'],'correct_neg'],
'CONDITIONAL': [['CONDITIONAL','TOPIC'],['COND-RHQ-Topic'],'correct_topic_or_cond'],
'TIME': [['CONDITIONAL','TOPIC'],['COND-RHQ-Topic'],'correct_topic_or_cond']
}


extra_error = {
'QUESTION_YN': [['QUESTION_WH'],['YN-WH'], 'error_ynq_sign_during_whq']
}


def operator_error_check(face_type_list,head_type_list,face_pred, head_pred,face_confidence_threshold, head_confidence_threshold, operator):
    face_pred_keys = [item[0] for item in face_pred if item[1] >= face_confidence_threshold]
    head_pred_keys = [item[0] for item in head_pred if item[1] >= head_confidence_threshold]
    face_overlap = [item for item in face_pred_keys if item in face_type_list ]
    head_overlap = [item for item in head_pred_keys if item in head_type_list]
    if operator == 'OR':
        if len(face_overlap)>0 or len(head_overlap)>0:
            return True
        else:
            return False
    if operator == 'AND':
        if len(face_overlap)>0 and len(head_overlap)>0:
            return True
        else:
            return False
    if operator == 'FACE':
        if len(face_overlap)>0:
            return True
        else:
            return False
    if operator == 'HEAD':
        if len(head_overlap)>0:
            return True
        else:
            return False


def begin_near_clause_boundary(clause_interval_list,pred_interval,clause_distance):
    if len(clause_interval_list)>0:
        distance = 1000000
        for clause_interval in clause_interval_list:
            d = pred_interval[0]-clause_interval[1]
            if d >=0 and d <= distance:
                distance = d
        if distance <= clause_distance:
            return True
        else:
            return False
    else:
        return False


def end_near_clause_boundary(clause_interval_list,pred_interval,clause_distance):
    if len(clause_interval_list)>0:
        distance = 1000000
        for clause_interval in clause_interval_list:
            d = clause_interval[0]-pred_interval[1]
            if d >=0 and d <= distance:
                distance = d
        if distance <= clause_distance:
            return True
        else:
            return False
    else:
        return False

def lexical_rules(gesture_based_video_output,clause_list):
    output_list=[]
    for item in gesture_based_video_output:
        error_check_flag = False
        (s,e),gesture_pred_I, face_pred, head_pred =item
        gesture_pred = gesture_pred_I[0]
        if gesture_pred_I[1]>= gesture_confidence:
            if gesture_pred == 'QUESTION_YN':
                face_filtered_high = [item[0] for item in face_pred if item[1] >= 80]
                face_filtered_low = [item[0] for item in face_pred if item[1] >= 0]
                head_filtered_low = [item[0] for item in head_pred if item[1] >= 0]
                if 'QUESTION_WH' in face_filtered_high:
                    error_message = 'error_ynq_sign_during_whq'
                else:    
                    if 'QUESTION_YN' in face_filtered_low or 'YN-WH' in head_filtered_low:
                        error_message = 'correct_ynq'
                    else:
                        error_message = 'error_ynq_lexical'
            else:                
                if gesture_pred in ['QUESTION_WH', 'NEGATIVE']:
                    error_check_flag = True
                elif gesture_pred in ['CONDITIONAL','TIME'] and begin_near_clause_boundary(clause_list,[s,e],clause_distance=clause_distance_lexical):
                    error_check_flag = True
                else:   
                    error_check_flag = False
                if error_check_flag:
                    face_type_list = lexical_error_dict[gesture_pred][0]
                    head_type_list = lexical_error_dict[gesture_pred][1]
                    if operator_error_check(face_type_list,head_type_list,face_pred, head_pred,face_confidence_threshold=face_confidence_lexical, head_confidence_threshold=head_confidence_lexical, operator = 'OR'):
                        error_message = lexical_correct_dict[gesture_pred][2]
                    else:
                        error_message = lexical_error_dict[gesture_pred][2]
                else:
                    error_message =''
            if len(error_message)>0:
                output_list.append([(s,e),error_message])
    return output_list


def begin_rules(face_based_video_output,clause_list):
    d = {'CONDITIONAL':'error_cond_beginning', 'TOPIC': 'error_topic_beginning','QUESTION_YN': 'error_ynq_beginning' }
    output_list=[]
    for item in face_based_video_output:
        face_interval, face_pred, head_pred =item  #face_pred : (face_class,face_confidence)
        s,e = face_interval
        face_pred_key , face_pred_score =  face_pred
        head_pred_keys = [item[0] for item in head_pred if item[1] >= 80]

        if face_pred_score>= 80 and not begin_near_clause_boundary(clause_list,[s,e], clause_distance= clause_distance_begin):
            error_check_flag = False

            if face_pred_key =='CONDITIONAL' and 'COND-RHQ-Topic' in head_pred_keys:
                error_check_flag = True
                
            elif face_pred_key == 'TOPIC'  and 'COND-RHQ-Topic' in head_pred_keys:
                error_check_flag = True

            elif face_pred_key =='QUESTION_YN' and 'YN-WH' in head_pred_keys:
                error_check_flag = True

            if error_check_flag:
                    error_message = d[face_pred_key]
                    output_list.append([(s,e),error_message])
    return output_list


def end_rules(face_based_video_output,clause_list):
    output_list=[]
    for item in face_based_video_output:
        face_interval, face_pred, head_pred =item  #face_pred : (face_class,face_confidence)
        s,e = face_interval
        face_pred_key , face_pred_score =  face_pred
        head_pred_keys = [item[0] for item in head_pred if item[1] >= 80]
        if face_pred_score>= 80 and not end_near_clause_boundary(clause_list,[s,e], clause_distance= clause_distance_end):
            if face_pred_key =='QUESTION_YN' and 'YN-WH' in head_pred_keys:
                output_list.append([(s,e),'error_ynq_end'])
    return output_list


def errors_post_processing(gesture_based_video_output, face_based_video_output, clause_list,org_name):
    error_dict_output={'Start':[],'End':[],'Error':[]}
    correct_dict_output={'Start':[],'End':[],'Correct':[]}
    output_list = lexical_rules(gesture_based_video_output,clause_list)+begin_rules(face_based_video_output,clause_list)+end_rules(face_based_video_output,clause_list)
    for line in output_list:
        (s,e),message = line
        if message.startswith('error'):
            error_dict_output['Start'].append(s)
            error_dict_output['End'].append(e)
            error_dict_output['Error'].append(message)
        elif message.startswith('correct'):
            correct_dict_output['Start'].append(s)
            correct_dict_output['End'].append(e)
            correct_dict_output['Correct'].append(message)
    ###################################################################
    error_output_list=[]
    included_message_list=[]
    d = dict(Counter(error_dict_output['Error']))
    l =sorted([(key,d[key]) for key in d.keys()], key=itemgetter(1), reverse= True)
    if len(l)>0:
        highest_freq = l[0][1]
        error_message_list=[item[0] for item in l if item[1]==highest_freq]
        for index in range(len(error_dict_output['Error'])):
            if error_dict_output['Error'][index] in error_message_list and error_dict_output['Error'][index] not in included_message_list:
                output_line = [error_dict_output[key][index] for key in ['Start', 'End','Error']]
                error_output_list.append(output_line)
                included_message_list.append(error_dict_output['Error'][index])
    ###################################################################
    correct_output_list=[]
    included_message_list=[]
    d = dict(Counter(correct_dict_output['Correct']))
    l =sorted([(key,d[key]) for key in d.keys()], key=itemgetter(1), reverse= True)
    if len(l)>0:
        highest_freq = l[0][1]
        error_message_list=[item[0] for item in l if item[1]==highest_freq]
        for index in range(len(correct_dict_output['Correct'])):
            if correct_dict_output['Correct'][index] in error_message_list and correct_dict_output['Correct'][index] not in included_message_list:
                output_line = [correct_dict_output[key][index] for key in ['Start', 'End','Correct']]
                correct_output_list.append(output_line)
                included_message_list.append(correct_dict_output['Correct'][index])

    ###################################################################
    return error_output_list+correct_output_list


def generate_error_file(gesture_based_video_output, face_based_video_output, clause_list,org_name):

    print('\n\n################################################')
    print('STEP 10: Generate the Feedback')
    print('################################################') 
        
    output_list = errors_post_processing(gesture_based_video_output, face_based_video_output, clause_list,org_name)

    csv_name = org_name[:-4] + '_feedback.csv'
    
    with open(csv_name, 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        
        for error_line in output_list:
            start, end, error_type= error_line
            error_message = message_dict[error_type]
            s = str(datetime.timedelta(seconds=round(start/30)))
            e = str(datetime.timedelta(seconds=round(end/30)))
            # print([s, e,error_type, error_message])
            spamwriter.writerow([s, e,error_type, error_message])

    TOKEN = 'FoQ2oS0Vu6AAAAAAAAAPqu6InIgnX3Jc7CoGsBVx_OkUtloKAH4X7yexKJwkEeLg'
    LOCALFILE = csv_name
    BACKUPPATH = csv_name[26:]
    dbx = dropbox.Dropbox(TOKEN)
    try:
        dbx.users_get_current_account()
    except AuthError as err:
        sys.exit("ERROR: Invalid access token; try re-generating an access token from the app console on the web.")
    with open(LOCALFILE, 'rb') as f:
        # We use WriteMode=overwrite to make sure that the settings in the file are changed on upload
        print("Uploading " + LOCALFILE + " to Dropbox as " + BACKUPPATH + "...")
        try:
            dbx.files_upload(f.read(), BACKUPPATH, mode=WriteMode('overwrite'))
        except ApiError as err:
            # This checks for the specific error where a user doesn't have enough Dropbox space quota to upload this file
            if (err.error.is_path() and
                    err.error.get_path().error.is_insufficient_space()):
                sys.exit("ERROR: Cannot back up; insufficient space.")
            elif err.user_message_text:
                print(err.user_message_text)
                sys.exit()
            else:
                print(err)
                sys.exit()


def camera_config_feedback(message,org_name):

    csv_name = org_name[:-4] + '_feedback.csv'
    
    with open(csv_name, 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([message])

    TOKEN = 'FoQ2oS0Vu6AAAAAAAAAPqu6InIgnX3Jc7CoGsBVx_OkUtloKAH4X7yexKJwkEeLg'
    LOCALFILE = csv_name
    BACKUPPATH = csv_name[26:]
    dbx = dropbox.Dropbox(TOKEN)
    try:
        dbx.users_get_current_account()
    except AuthError as err:
        sys.exit("ERROR: Invalid access token; try re-generating an access token from the app console on the web.")
    with open(LOCALFILE, 'rb') as f:
        # We use WriteMode=overwrite to make sure that the settings in the file are changed on upload
        print("Uploading " + LOCALFILE + " to Dropbox as " + BACKUPPATH + "...")
        try:
            dbx.files_upload(f.read(), BACKUPPATH, mode=WriteMode('overwrite'))
        except ApiError as err:
            # This checks for the specific error where a user doesn't have enough Dropbox space quota to upload this file
            if (err.error.is_path() and
                    err.error.get_path().error.is_insufficient_space()):
                sys.exit("ERROR: Cannot back up; insufficient space.")
            elif err.user_message_text:
                print(err.user_message_text)
                sys.exit()
            else:
                print(err)
                sys.exit()


def generate_error_file_locally(gesture_based_video_output, face_based_video_output, clause_list,org_name):

    print('\n\n################################################')
    print('STEP 10: Generate the Feedback')
    print('################################################') 
        
    output_list = errors_post_processing(gesture_based_video_output, face_based_video_output, clause_list,org_name)

    csv_name = org_name[:-4] + '_feedback.csv'
    
    with open(csv_name, 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        
        for error_line in output_list:
            start, end, error_type= error_line
            error_message = message_dict[error_type]
            s = str(datetime.timedelta(seconds=round(start/30)))
            e = str(datetime.timedelta(seconds=round(end/30)))
            # print([s, e,error_type, error_message])
            spamwriter.writerow([s, e,error_type, error_message])