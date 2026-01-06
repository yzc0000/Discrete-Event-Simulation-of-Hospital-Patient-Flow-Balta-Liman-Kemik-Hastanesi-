import simpy
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys
import datetime

sys.setrecursionlimit(2000)

# ----------- Configuration -----------
SIM_TIME = 480
WALKIN_CUTOFF_TIME = SIM_TIME - 60
LUNCH_START = 240       
LUNCH_END = 300
NUM_DOCTORS = 7
NUM_XRAY_ROOMS = 2
NUM_CONSULTANTS = 1
RANDOM_SEED = 42
QUEUE_TRACK_INTERVAL = 1
AFTERNOON_SPEEDUP_FACTOR = 0.85

UNIFORM_MIN_DEVIATION_MINUTES = -5
UNIFORM_MAX_DEVIATION_MINUTES = 10

WALKIN_PRIORITY_OFFSET = SIM_TIME * 1000

APPOINTMENT_ONLY_DOCTOR_ID = 0
WALKIN_ONLY_DOCTOR_ID = 1


CONSULTANT_SERVICE_TIME_MIN = 2- (6 ** 0.5) / 2  # ≈ 20.515
CONSULTANT_SERVICE_TIME_MAX = 2 + (6 ** 0.5) / 2  # ≈ 39.485


def consultant_service_time():
    return random.uniform(CONSULTANT_SERVICE_TIME_MIN, CONSULTANT_SERVICE_TIME_MAX)


random.seed(RANDOM_SEED)

xray_resources = []
doctors = []
consultant_resources = []

# Doctor configurations (Uniform appointment arrival)
doctor_configs = [
    {  # Doktor 1 (ID 0)
        "arrival": lambda: random.expovariate(1 / 10),
        "type_a_first_exam": lambda: max(random.gammavariate(1.15, 2.41), 0.25),
        "type_b_first_exam": lambda: max(random.gammavariate(0.80, 4.71), 0.70),
        "type_a_second_exam": lambda: max(random.gammavariate(2.3, 1.5), 0.25),
        "appointment_interval": lambda: max(random.uniform(9, 15), 10),
        "xray_probability": 0.76
    },
    {  # Doktor 2 (ID 1)
        "arrival": lambda: random.expovariate(1 / 12),
        "type_a_first_exam": lambda: max(random.gammavariate(3, 1), 0.5),
        "type_b_first_exam": lambda: max(random.gammavariate(3, 1), 0.5),
        "type_a_second_exam": lambda: max(random.gammavariate(4, 1), 0.5),
        "appointment_interval": lambda: max(random.uniform(13, 17), 1),
        "xray_probability": 0.80
    },
    {  # Doktor 3 (ID 2)
        "arrival": lambda: random.expovariate(1 / 12),
        "type_a_first_exam": lambda: max(random.gammavariate(2, 1), 0.5),
        "type_b_first_exam": lambda: max(random.gammavariate(4, 1), 0.5),
        "type_a_second_exam": lambda: max(random.gammavariate(6, 1), 0.5),
        "appointment_interval": lambda: max(random.uniform(11, 15), 1),
        "xray_probability": 0.78
    },
    {  # Doktor 4 (ID 3)
        "arrival": lambda: random.expovariate(1 / 11),
        "type_a_first_exam": lambda: max(random.gammavariate(2, 1), 0.5),
        "type_b_first_exam": lambda: max(random.gammavariate(3, 1), 0.5),
        "type_a_second_exam": lambda: max(random.gammavariate(4, 1), 0.5),
        "appointment_interval": lambda: max(random.uniform(8, 12), 1),
        "xray_probability": 0.60
    },
    {  # Doktor 5 (ID 4)
        "arrival": lambda: random.expovariate(1 / 13),
        "type_a_first_exam": lambda: max(random.gammavariate(2, 1), 0.5),
        "type_b_first_exam": lambda: max(random.gammavariate(3, 1), 0.5),
        "type_a_second_exam": lambda: max(random.gammavariate(5, 1), 0.5),
        "appointment_interval": lambda: max(random.uniform(9, 13), 1),
        "xray_probability": 0.65
    },
    {  # Doktor 6 (ID 5)
        "arrival": lambda: random.expovariate(1 / 10),
        "type_a_first_exam": lambda: max(random.gammavariate(2, 1), 0.5),
        "type_b_first_exam": lambda: max(random.gammavariate(5, 1), 0.5),
        "type_a_second_exam": lambda: max(random.gammavariate(2, 1), 0.5),
        "appointment_interval": lambda: max(random.uniform(11, 17), 1),
        "xray_probability": 0.90
    },
    {  # Doktor 7 (ID 6)
        "arrival": lambda: random.expovariate(1 / 14),
        "type_a_first_exam": lambda: max(random.gammavariate(2, 1), 0.5),
        "type_b_first_exam": lambda: max(random.gammavariate(5, 1), 0.5),
        "type_a_second_exam": lambda: max(random.gammavariate(3, 1), 0.5),
        "appointment_interval": lambda: max(random.uniform(10, 16), 1),
        "xray_probability": 0.87
    }
]
base_xray_service_time = lambda: max(random.gammavariate(1.25, 1.9), 1)


def get_actual_xray_service_time(env_now_time):
    service_time = base_xray_service_time()
    if env_now_time < LUNCH_END:
        return service_time * 1.25
    else:
        return service_time


# ----------- Statistics Tracking -----------
queue_lengths = defaultdict(list)
xray_room_queue_lengths = defaultdict(list)
consultant_queue_lengths = defaultdict(list)
timestamps = []
doctor_patient_count = [0] * NUM_DOCTORS
xray_patient_count = 0
consultant_patient_count = [0] * NUM_CONSULTANTS
doctor_second_exam_count = [0] * NUM_DOCTORS
appointment_arrival_count = 0
appointment_departure_count = 0
walk_in_arrival_count = 0
walk_in_departure_count = 0
total_patients_generated = 0
patients_currently_in_system = 0
appointment_actual_arrival_times = defaultdict(list)
appointment_scheduled_times = defaultdict(list)

# ----------- Doctor Lunch State -----------
doctor_is_on_lunch_break = [False] * NUM_DOCTORS


# ----------- Simulation Processes -----------

def manage_doctor_lunch_state(env, doctor_id):
    global doctor_is_on_lunch_break
    yield env.timeout(max(0, LUNCH_START - env.now))
    print(f"--- {env.now:.2f} - Dr {doctor_id + 1} ÖĞLE MOLASI PERİYODU BAŞLADI (mevcut hastayı bitirecek) ---")
    doctor_is_on_lunch_break[doctor_id] = True
    yield env.timeout(max(0, LUNCH_END - env.now))
    print(f"--- {env.now:.2f} - Dr {doctor_id + 1} ÖĞLE MOLASI PERİYODU BİTTİ (hizmete devam) ---")
    doctor_is_on_lunch_break[doctor_id] = False


def patient(env, name, doctor_id, doctor_resource, is_appointment, needs_xray,
            actual_arrival_time, scheduled_arrival_time_for_priority=None):
    global doctor_patient_count, xray_patient_count, doctor_second_exam_count
    global appointment_departure_count, walk_in_departure_count
    global patients_currently_in_system, doctor_is_on_lunch_break
    global xray_resources, consultant_resources, consultant_patient_count

    if is_appointment:
        request_priority = scheduled_arrival_time_for_priority
        type_str_detail = f"Sched@{scheduled_arrival_time_for_priority:.2f}"
    else:
        request_priority = WALKIN_PRIORITY_OFFSET + actual_arrival_time
        type_str_detail = "Walk-in"

    type_str_base = f"{'Randevulu' if is_appointment else 'Randevusuz'} Tip-{'A' if needs_xray else 'B'}"
    print(f"{actual_arrival_time:.2f} - {name} ({type_str_base}, {type_str_detail}) GELDI Dr {doctor_id + 1}.")

    try:
        if not is_appointment:

            consultant_queue_lengths_current = [len(c.queue) for c in consultant_resources]
            min_consultant_queue = min(consultant_queue_lengths_current)
            available_consultants = [i for i, q_len in enumerate(consultant_queue_lengths_current)
                                     if q_len == min_consultant_queue]
            selected_consultant_idx = random.choice(available_consultants)
            selected_consultant = consultant_resources[selected_consultant_idx]

            print(
                f"{env.now:.2f} - {name} Danışman {selected_consultant_idx + 1} talep ediyor (Kayıt için). Kuyruk: {len(selected_consultant.queue)}")
            start_wait_consultant = env.now
            with selected_consultant.request() as req_consultant:
                try:
                    yield req_consultant
                    wait_time_consultant = env.now - start_wait_consultant
                    print(
                        f"{env.now:.2f} - {name} Danışman {selected_consultant_idx + 1} ile kayıt BAŞLADI. Bekleme: {wait_time_consultant:.2f}dk.")

                    registration_time = consultant_service_time()
                    yield env.timeout(registration_time)
                    consultant_patient_count[selected_consultant_idx] += 1
                    print(
                        f"{env.now:.2f} - {name} Danışman {selected_consultant_idx + 1} kayıt BİTTİ (Süre: {registration_time:.2f}dk).")
                except simpy.Interrupt as interrupt:
                    print(
                        f"{env.now:.2f} - {name} ({type_str_base}) KESİLDİ (Danışman kayıt) {selected_consultant_idx + 1}: {interrupt.cause}")
                    return

        # First Examination
        if doctor_is_on_lunch_break[doctor_id] and LUNCH_START <= env.now < LUNCH_END:
            if actual_arrival_time < LUNCH_END:
                wait_duration = LUNCH_END - env.now
                if wait_duration > 0:
                    print(
                        f"{env.now:.2f} - {name} Dr {doctor_id + 1} için öğle molası bitimini ({LUNCH_END:.2f}) bekliyor (1. muayene öncesi). Kalan: {wait_duration:.2f}dk.")
                    yield env.timeout(wait_duration)
                    print(
                        f"{env.now:.2f} - {name} Dr {doctor_id + 1} için öğle molası sonrası devam ediyor (1. muayene).")

        print(
            f"{env.now:.2f} - {name} Dr {doctor_id + 1} talep ediyor (Prio: {request_priority:.2f}). Kuyruk: {len(doctor_resource.queue)}")
        start_wait_doc1 = env.now
        with doctor_resource.request(priority=request_priority) as req:
            try:
                yield req
                wait_time_doc1 = env.now - start_wait_doc1
                print(
                    f"{env.now:.2f} - {name} 1. muayene BAŞLADI Dr {doctor_id + 1}. Bekleme: {wait_time_doc1:.2f}dk. (Talep Prio: {request_priority:.2f})")

                service_start_time_doc1 = env.now
                base_exam_time = doctor_configs[doctor_id]["type_a_first_exam"]() if needs_xray else \
                doctor_configs[doctor_id]["type_b_first_exam"]()
                exam_time = base_exam_time
                speed_up_applied_doc1 = False
                if service_start_time_doc1 >= LUNCH_END:
                    exam_time = base_exam_time * AFTERNOON_SPEEDUP_FACTOR
                    speed_up_applied_doc1 = True

                yield env.timeout(exam_time)
                doctor_patient_count[doctor_id] += 1
                print(
                    f"{env.now:.2f} - {name} 1. muayene BİTTİ Dr {doctor_id + 1} (Süre: {exam_time:.2f}dk {'[Hızlandırıldı]' if speed_up_applied_doc1 else ''}).")
            except simpy.Interrupt as interrupt:
                print(
                    f"{env.now:.2f} - {name} ({type_str_base}) KESİLDİ (1. muayene) Dr {doctor_id + 1}: {interrupt.cause}")
                return

        # X-ray Process
        if needs_xray:
            xray_priority = request_priority
            print(f"{env.now:.2f} - {name} X-ray için uygun oda arıyor (Prio: {xray_priority:.2f}).")

            room_queue_lengths = [len(xr.queue) for xr in xray_resources]
            min_queue_len = min(room_queue_lengths)
            candidate_room_indices = [i for i, q_len in enumerate(room_queue_lengths) if q_len == min_queue_len]
            selected_xray_room_idx = random.choice(candidate_room_indices)
            chosen_xray_resource = xray_resources[selected_xray_room_idx]

            print(
                f"{env.now:.2f} - {name} X-ray Oda {selected_xray_room_idx + 1} talep ediyor. Oda Kuyruğu: {len(chosen_xray_resource.queue)}")
            start_wait_xray = env.now
            with chosen_xray_resource.request(priority=xray_priority) as req_xray:
                try:
                    yield req_xray
                    wait_time_xray = env.now - start_wait_xray
                    print(
                        f"{env.now:.2f} - {name} X-ray Oda {selected_xray_room_idx + 1} BAŞLADI. Bekleme: {wait_time_xray:.2f}dk.")

                    xray_time_val = get_actual_xray_service_time(env.now)
                    yield env.timeout(xray_time_val)
                    xray_patient_count += 1
                    print(
                        f"{env.now:.2f} - {name} X-ray Oda {selected_xray_room_idx + 1} BİTTİ (Süre: {xray_time_val:.2f}dk).")
                except simpy.Interrupt as interrupt:
                    print(
                        f"{env.now:.2f} - {name} ({type_str_base}) KESİLDİ (X-ray Oda {selected_xray_room_idx + 1}): {interrupt.cause}")
                    return

            # Second Examination
            second_exam_priority = request_priority
            if doctor_is_on_lunch_break[doctor_id] and LUNCH_START <= env.now < LUNCH_END:
                if actual_arrival_time < LUNCH_END:
                    wait_duration_doc2 = LUNCH_END - env.now
                    if wait_duration_doc2 > 0:
                        print(
                            f"{env.now:.2f} - {name} Dr {doctor_id + 1} için öğle molası bitimini ({LUNCH_END:.2f}) bekliyor (2. muayene öncesi). Kalan: {wait_duration_doc2:.2f}dk.")
                        yield env.timeout(wait_duration_doc2)
                        print(
                            f"{env.now:.2f} - {name} Dr {doctor_id + 1} için öğle molası sonrası devam ediyor (2. muayene).")

            print(
                f"{env.now:.2f} - {name} Dr {doctor_id + 1} 2. muayene talep ediyor (Prio: {second_exam_priority:.2f}). Kuyruk: {len(doctor_resource.queue)}")
            start_wait_doc2 = env.now
            with doctor_resource.request(priority=second_exam_priority) as req_doc2:
                try:
                    yield req_doc2
                    wait_time_doc2 = env.now - start_wait_doc2
                    print(
                        f"{env.now:.2f} - {name} 2. muayene BAŞLADI Dr {doctor_id + 1}. Bekleme: {wait_time_doc2:.2f}dk.")

                    service_start_time_doc2 = env.now
                    base_second_exam_time = doctor_configs[doctor_id]["type_a_second_exam"]()
                    second_exam_time = base_second_exam_time
                    speed_up_applied_doc2 = False
                    if service_start_time_doc2 >= LUNCH_END:
                        second_exam_time = base_second_exam_time * AFTERNOON_SPEEDUP_FACTOR
                        speed_up_applied_doc2 = True

                    yield env.timeout(second_exam_time)
                    doctor_second_exam_count[doctor_id] += 1
                    print(
                        f"{env.now:.2f} - {name} 2. muayene BİTTİ Dr {doctor_id + 1} (Süre: {second_exam_time:.2f}dk {'[Hızlandırıldı]' if speed_up_applied_doc2 else ''}).")
                except simpy.Interrupt as interrupt:
                    print(
                        f"{env.now:.2f} - {name} ({type_str_base}) KESİLDİ (2. muayene) Dr {doctor_id + 1}: {interrupt.cause}")
                    return
        # Departure
        departure_time = env.now
        if is_appointment:
            appointment_departure_count += 1
        else:
            walk_in_departure_count += 1
        print(
            f"{departure_time:.2f} - {name} AYRILDI. Sistemde kalma süresi: {departure_time - actual_arrival_time:.2f} dk.")
    finally:
        patients_currently_in_system -= 1
        if patients_currently_in_system < 0:
            print(f"!!!! HATA !!!! Hasta sayısı {env.now:.2f} anında {name} için sıfırın altına düştü!")
            patients_currently_in_system = 0


def patient_generator(env, doctor_id, doctor_resource):
    global total_patients_generated, walk_in_arrival_count, patients_currently_in_system

    if APPOINTMENT_ONLY_DOCTOR_ID is not None and doctor_id == APPOINTMENT_ONLY_DOCTOR_ID:
        print(
            f"--- Dr {doctor_id + 1} SADECE RANDEVULU hasta baktığı için randevusuz (walk-in) hasta üreticisi bu doktor için başlatılmadı. ---")
        return

    patient_idx_walkin = 0
    while True:
        interarrival_time = doctor_configs[doctor_id]["arrival"]()
        potential_next_arrival = env.now + interarrival_time
        if potential_next_arrival >= WALKIN_CUTOFF_TIME:
            print(
                f"{env.now:.2f} - Dr {doctor_id + 1} Walk-in Üreticisi DURUYOR. Bir sonraki geliş ({potential_next_arrival:.2f}) cutoff ({WALKIN_CUTOFF_TIME}) zamanını aşacaktı.")
            break
        yield env.timeout(max(0, interarrival_time))
        actual_arrival_time = env.now
        if actual_arrival_time >= WALKIN_CUTOFF_TIME:
            break
        patient_idx_walkin += 1
        total_patients_generated += 1
        walk_in_arrival_count += 1
        patients_currently_in_system += 1
        is_xray_needed = random.random() < doctor_configs[doctor_id]["xray_probability"]
        patient_name = f"Hasta-RZ-{doctor_id + 1}-{patient_idx_walkin}"
        env.process(patient(env, patient_name, doctor_id, doctor_resource, False, is_xray_needed,
                            actual_arrival_time, scheduled_arrival_time_for_priority=None))


def appointment_generator(env, doctor_id, doctor_resource, scheduled_times_for_this_doctor):
    global total_patients_generated, appointment_arrival_count, patients_currently_in_system
    global appointment_actual_arrival_times

    if WALKIN_ONLY_DOCTOR_ID is not None and doctor_id == WALKIN_ONLY_DOCTOR_ID:
        print(
            f"--- Dr {doctor_id + 1} SADECE RANDEVUSUZ hasta baktığı için randevulu hasta üreticisi bu doktor için başlatılmadı. ---")
        return

    patient_idx_appt = 0
    for scheduled_time in scheduled_times_for_this_doctor:
        patient_idx_appt += 1
        punctuality_deviation = random.uniform(UNIFORM_MIN_DEVIATION_MINUTES, UNIFORM_MAX_DEVIATION_MINUTES)
        actual_arrival_time_candidate = max(0, scheduled_time + punctuality_deviation)

        delay_until_actual_arrival = actual_arrival_time_candidate - env.now
        if delay_until_actual_arrival > 0:
            yield env.timeout(delay_until_actual_arrival)

        current_actual_arrival_time = env.now

        if current_actual_arrival_time >= SIM_TIME:
            continue

        appointment_actual_arrival_times[doctor_id].append(current_actual_arrival_time)
        total_patients_generated += 1
        appointment_arrival_count += 1
        patients_currently_in_system += 1
        is_xray_needed = random.random() < doctor_configs[doctor_id]["xray_probability"]
        patient_name = f"Hasta-RD-{doctor_id + 1}-{patient_idx_appt}"
        env.process(patient(env, patient_name, doctor_id, doctor_resource, True, is_xray_needed,
                            current_actual_arrival_time, scheduled_arrival_time_for_priority=scheduled_time))

def track_queues(env, doctors_list, stop_event_tracker):
    global timestamps, xray_resources, xray_room_queue_lengths, consultant_resources, consultant_queue_lengths
    while not stop_event_tracker.triggered:
        current_time_track = env.now
        timestamps.append(current_time_track)
        for i, d_res in enumerate(doctors_list):
            queue_lengths[i].append(len(d_res.queue))

        for room_idx in range(NUM_XRAY_ROOMS):
            xray_room_queue_lengths[room_idx].append(len(xray_resources[room_idx].queue))

        for consultant_idx in range(NUM_CONSULTANTS):
            consultant_queue_lengths[consultant_idx].append(len(consultant_resources[consultant_idx].queue))

        start_wait_track = env.now
        timeout_duration = QUEUE_TRACK_INTERVAL
        while timeout_duration > 0 and not stop_event_tracker.triggered:
            yield env.timeout(min(timeout_duration, 0.5))
            timeout_duration = QUEUE_TRACK_INTERVAL - (env.now - start_wait_track)

    current_time_track = env.now
    timestamps.append(current_time_track)
    for i, d_res in enumerate(doctors_list):
        queue_lengths[i].append(len(d_res.queue))
    for room_idx in range(NUM_XRAY_ROOMS):
        xray_room_queue_lengths[room_idx].append(len(xray_resources[room_idx].queue))

    for consultant_idx in range(NUM_CONSULTANTS):
        consultant_queue_lengths[consultant_idx].append(len(consultant_resources[consultant_idx].queue))
    print(f"{env.now:.2f} - Kuyruk takibi durdu.")


def simulation_ender(env, stop_event_obj, doctors_list_for_check):
    global xray_resources, consultant_resources
    if env.now < SIM_TIME:
        yield env.timeout(SIM_TIME - env.now)

    print(
        f"--- {env.now:.2f} - RANDEVU KESİM ({SIM_TIME}dk) zamanına ulaşıldı. Walk-in'ler {WALKIN_CUTOFF_TIME}dk'da durdu. Mevcut hastalar tamamlanıyor. ---")

    check_interval = 2
    while True:
        if patients_currently_in_system <= 0:
            all_doc_queues_empty = all(len(d.queue) == 0 for d in doctors_list_for_check)
            all_doc_resources_free = all(d.count == 0 for d in doctors_list_for_check)

            all_xray_queues_empty = all(len(xr.queue) == 0 for xr in xray_resources)
            all_xray_resources_free = all(xr.count == 0 for xr in xray_resources)

            all_consultant_queues_empty = all(len(cr.queue) == 0 for cr in consultant_resources)
            all_consultant_resources_free = all(cr.count == 0 for cr in consultant_resources)

            if (all_doc_queues_empty and all_doc_resources_free and
                    all_xray_queues_empty and all_xray_resources_free and
                    all_consultant_queues_empty and all_consultant_resources_free):
                print(
                    f"--- {env.now:.2f} - SİSTEM BOŞ (Hasta Sayacı={patients_currently_in_system}, Tüm Kuyruklar ve Kaynaklar boş). Simülasyon durduruluyor. ---")
                if not stop_event_obj.triggered:
                    stop_event_obj.succeed()
                break
            else:
                if int(env.now) % (check_interval * 5) < check_interval:
                    xray_q_details = [len(xr.queue) for xr in xray_resources]
                    xray_u_details = [xr.count for xr in xray_resources]
                    consultant_q_details = [len(cr.queue) for cr in consultant_resources]
                    consultant_u_details = [cr.count for cr in consultant_resources]
                    print(
                        f"UYARI: {env.now:.2f} - Hasta sayacı={patients_currently_in_system} ancak kaynaklar/kuyruklar boş değil. Tekrar kontrol ediliyor. DocQ: {[len(d.queue) for d in doctors_list_for_check]}, XrayQs: {xray_q_details}, ConsultantQs: {consultant_q_details}, DocUsers: {[d.count for d in doctors_list_for_check]}, XrayUsers: {xray_u_details}, ConsultantUsers: {consultant_u_details}")

        elif env.now > SIM_TIME + 3 * SIM_TIME:
            print(
                f"--- {env.now:.2f} - UZUN SÜRE UYARISI. Hasta sayacı={patients_currently_in_system}. Simülasyon zorla durduruluyor. ---")
            if not stop_event_obj.triggered:
                stop_event_obj.succeed()
            break
        yield env.timeout(check_interval)


def descriptive_stats(data, label):
    if not data:
        print(f"\n{label} Tanımlayıcı İstatistikler: (Veri Yok)")
        return
    data_array = np.array(data)
    print(f"\n{label} Tanımlayıcı İstatistikler:")
    print(f"  Ortalama: {np.mean(data_array):.2f}")
    print(f"  Medyan: {np.median(data_array):.2f}")
    print(f"  Standart Sapma: {np.std(data_array):.2f}")
    print(f"  Min: {np.min(data_array):.2f}")
    print(f"  Maks: {np.max(data_array):.2f}")
    max_val = np.max(data_array)
    if max_val > 0:
        print(f"  Zamanın Yüzdesi > 0: {100 * np.mean(data_array > 0):.2f}%")
    else:
        print(f"  Zamanın Yüzdesi > 0: 0.00%")


# ----------- Main Simulation Execution -----------
def main():
    global APPOINTMENT_ONLY_DOCTOR_ID
    global WALKIN_ONLY_DOCTOR_ID

    start_real_time = datetime.datetime.now()
    print(f"Simülasyon Başlatılıyor - Randevu Kesim: {SIM_TIME} dak, Walk-in Kesim: {WALKIN_CUTOFF_TIME} dak")
    print(f"Öğle Yemeği Molası PERİYODU: {LUNCH_START} - {LUNCH_END} dakika")
    print(
        f"ÖĞLEDEN SONRA HIZLANDIRMA (Doktorlar): Doktor hizmet süreleri {AFTERNOON_SPEEDUP_FACTOR:.0%} ile çarpılacak (zaman >= {LUNCH_END})")
    print(f"Doktor Sayısı: {NUM_DOCTORS}, X-ray Oda Sayısı: {NUM_XRAY_ROOMS}")
    print(f"Danışman Sayısı: {NUM_CONSULTANTS} (Randevusuz hastalar için kayıt)")
    print(
        f"Danışman Hizmet Süresi: Uniform dağılım (Ortalama=30dk, Varyans=30, Aralık=[{CONSULTANT_SERVICE_TIME_MIN:.2f}, {CONSULTANT_SERVICE_TIME_MAX:.2f}])")
    print(f"X-ray Servis Süresi: Öğleden önce {1.25:.2f}x yavaş, öğleden sonra normal.")
    print(
        f"Randevu Dakikliği (Uniform Dağılım): Min Sapma={UNIFORM_MIN_DEVIATION_MINUTES}dk, Maks Sapma={UNIFORM_MAX_DEVIATION_MINUTES}dk")
    print(f"Rastgele Tohum: {RANDOM_SEED}")

    if APPOINTMENT_ONLY_DOCTOR_ID is not None:
        if 0 <= APPOINTMENT_ONLY_DOCTOR_ID < NUM_DOCTORS:
            print(f"ÖZEL ROL: Doktor {APPOINTMENT_ONLY_DOCTOR_ID + 1} SADECE RANDEVULU hasta bakacak.")
        else:
            print(
                f"UYARI: APPOINTMENT_ONLY_DOCTOR_ID ({APPOINTMENT_ONLY_DOCTOR_ID}) geçersiz. Bu özel rol devre dışı bırakılacak.")
            APPOINTMENT_ONLY_DOCTOR_ID = None
    if WALKIN_ONLY_DOCTOR_ID is not None:
        if 0 <= WALKIN_ONLY_DOCTOR_ID < NUM_DOCTORS:
            print(f"ÖZEL ROL: Do    ktor {WALKIN_ONLY_DOCTOR_ID + 1} SADECE RANDEVUSUZ (walk-in) hasta bakacak.")
        else:
            print(
                f"UYARI: WALKIN_ONLY_DOCTOR_ID ({WALKIN_ONLY_DOCTOR_ID}) geçersiz. Bu özel rol devre dışı bırakılacak.")
            WALKIN_ONLY_DOCTOR_ID = None
    if APPOINTMENT_ONLY_DOCTOR_ID is not None and APPOINTMENT_ONLY_DOCTOR_ID == WALKIN_ONLY_DOCTOR_ID:
        print(
            f"UYARI: APPOINTMENT_ONLY_DOCTOR_ID ve WALKIN_ONLY_DOCTOR_ID aynı doktora ({APPOINTMENT_ONLY_DOCTOR_ID + 1}) atanmış. Bu doktor hiç hasta göremeyebilir. Lütfen ID'leri düzeltin.")

    global doctors, xray_resources, consultant_resources, doctor_is_on_lunch_break
    global queue_lengths, xray_room_queue_lengths, consultant_queue_lengths, timestamps
    global doctor_patient_count, xray_patient_count, consultant_patient_count, doctor_second_exam_count
    global appointment_arrival_count, appointment_departure_count
    global walk_in_arrival_count, walk_in_departure_count
    global total_patients_generated, patients_currently_in_system
    global appointment_actual_arrival_times, appointment_scheduled_times

    queue_lengths.clear();
    xray_room_queue_lengths.clear();
    consultant_queue_lengths.clear();
    timestamps.clear()
    appointment_actual_arrival_times.clear();
    appointment_scheduled_times.clear()
    doctor_patient_count = [0] * NUM_DOCTORS;
    xray_patient_count = 0
    consultant_patient_count = [0] * NUM_CONSULTANTS
    doctor_second_exam_count = [0] * NUM_DOCTORS;
    appointment_arrival_count = 0
    appointment_departure_count = 0;
    walk_in_arrival_count = 0
    walk_in_departure_count = 0;
    total_patients_generated = 0
    patients_currently_in_system = 0
    doctor_is_on_lunch_break = [False] * NUM_DOCTORS
    xray_resources = [];
    consultant_resources = []

    all_doctors_schedules = []
    print("\n--- Planlanan Randevu Saatleri Oluşturuluyor ---")
    for i in range(NUM_DOCTORS):
        if WALKIN_ONLY_DOCTOR_ID is not None and i == WALKIN_ONLY_DOCTOR_ID:
            all_doctors_schedules.append([])
            appointment_scheduled_times[i] = []
            print(f"  Dr {i + 1} (Sadece Randevusuz) için randevu planlanmadı.")
            continue

        doctor_schedule = [];
        current_scheduled_time = 0
        while True:
            interval = doctor_configs[i]["appointment_interval"]()
            next_scheduled_time = current_scheduled_time + interval
            if next_scheduled_time < SIM_TIME:
                doctor_schedule.append(next_scheduled_time)
                current_scheduled_time = next_scheduled_time
            else:
                break
        all_doctors_schedules.append(doctor_schedule)
        appointment_scheduled_times[i] = doctor_schedule
        last_sched_time_str = f"{doctor_schedule[-1]:.2f}" if doctor_schedule else "Yok"
        print(
            f"  Dr {i + 1} için {len(doctor_schedule)} randevu planlandı. Son planlanan: {last_sched_time_str} (Limit: {SIM_TIME})")
    print("--- Randevu Planlama Bitti ---\n")

    env = simpy.Environment()
    stop_event = env.event()

    xray_resources.extend([simpy.PriorityResource(env, capacity=1) for _ in range(NUM_XRAY_ROOMS)])
    consultant_resources.extend([simpy.Resource(env, capacity=1) for _ in range(NUM_CONSULTANTS)])
    doctors = [simpy.PriorityResource(env, capacity=1) for _ in range(NUM_DOCTORS)]

    print("--- Hasta Üreteçleri Başlatılıyor ---")
    for i in range(NUM_DOCTORS):
        doctor_res = doctors[i]

        is_appointment_only = (APPOINTMENT_ONLY_DOCTOR_ID is not None and i == APPOINTMENT_ONLY_DOCTOR_ID)
        is_walkin_only = (WALKIN_ONLY_DOCTOR_ID is not None and i == WALKIN_ONLY_DOCTOR_ID)

        if is_appointment_only:
            if all_doctors_schedules[i]:
                env.process(appointment_generator(env, i, doctor_res, all_doctors_schedules[i]))
                print(f"  Dr {i + 1} için SADECE RANDEVULU hasta üreteci başlatıldı.")
            else:
                print(f"  Dr {i + 1} (Sadece Randevulu) için planlanmış randevu yok, randevu üreteci başlatılmadı.")
        elif is_walkin_only:
            env.process(patient_generator(env, i, doctor_res))
            print(f"  Dr {i + 1} için SADECE RANDEVUSUZ hasta üreteci başlatıldı.")
        else:
            env.process(patient_generator(env, i, doctor_res))
            if all_doctors_schedules[i]:
                env.process(appointment_generator(env, i, doctor_res, all_doctors_schedules[i]))
                print(f"  Dr {i + 1} için HEM RANDEVULU HEM RANDEVUSUZ hasta üreteçleri başlatıldı.")
            else:
                print(f"  Dr {i + 1} için RANDEVUSUZ hasta üreteci başlatıldı (planlanmış randevu yok).")

        env.process(manage_doctor_lunch_state(env, i))

    print(f"--- {NUM_CONSULTANTS} Danışman kaynağı başlatıldı (Randevusuz hastalar için kayıt) ---")

    env.process(track_queues(env, doctors, stop_event))
    env.process(simulation_ender(env, stop_event, doctors))
    env.run(until=stop_event)

    end_real_time = datetime.datetime.now()
    print(f"\n--- Simülasyon Bitti ---")
    print(f"Toplam Simülasyon Süresi: {env.now:.2f} dakika")
    print(f"Gerçek Çalışma Süresi: {end_real_time - start_real_time}")

    min_len_ts = len(timestamps)
    if not timestamps or min_len_ts < 2:
        print("\nUYARI: Grafik çizimi için yeterli zaman damgası verisi yok.")
    else:
        plot_timestamps = np.array(timestamps)

        # Doctor Queue Plot
        plt.figure(figsize=(14, 7))
        for i in range(NUM_DOCTORS):
            q_data = queue_lengths.get(i, [])
            if len(q_data) < min_len_ts:
                last_val = q_data[-1] if q_data else 0
                q_data.extend([last_val] * (min_len_ts - len(q_data)))
            elif len(q_data) > min_len_ts:
                q_data = q_data[:min_len_ts]
            plt.plot(plot_timestamps, q_data, label=f'Dr {i + 1} Kuyruk', alpha=0.8)

        plt.axvline(x=LUNCH_START, color='grey', linestyle=':', linewidth=1, label=f'Öğle Başlangıcı ({LUNCH_START})')
        plt.axvline(x=LUNCH_END, color='dimgrey', linestyle='--', linewidth=1.2,
                    label=f'Öğle Bitişi / Hızlanma ({LUNCH_END})')
        plt.axvline(x=WALKIN_CUTOFF_TIME, color='blue', linestyle='-.', linewidth=1.2,
                    label=f'Walk-in Kesim ({WALKIN_CUTOFF_TIME})')
        plt.axvline(x=SIM_TIME, color='red', linestyle='-', linewidth=1.5, label=f'Randevu Kesim ({SIM_TIME})')
        title_suffix = ""
        if APPOINTMENT_ONLY_DOCTOR_ID is not None: title_suffix += f"\nDr {APPOINTMENT_ONLY_DOCTOR_ID + 1} Sadece Randevulu"
        if WALKIN_ONLY_DOCTOR_ID is not None: title_suffix += f", Dr {WALKIN_ONLY_DOCTOR_ID + 1} Sadece Randevusuz"

        plt.xlabel("Zaman (dakika)");
        plt.ylabel("Kuyruk Uzunluğu")
        plt.title(
            f"Doktor Kuyruk Uzunlukları (Punctuality: Uniform [{UNIFORM_MIN_DEVIATION_MINUTES},{UNIFORM_MAX_DEVIATION_MINUTES}])\nTohum: {RANDOM_SEED}{title_suffix}")
        plt.legend(fontsize='small', loc='upper left');
        plt.grid(True, linestyle=':', alpha=0.7);
        plt.tight_layout();
        plt.show()

        # X-ray Queue Plot
        plt.figure(figsize=(14, 7))
        for room_idx in range(NUM_XRAY_ROOMS):
            xray_data_plot = xray_room_queue_lengths.get(room_idx, [])
            if len(xray_data_plot) < min_len_ts:
                last_val = xray_data_plot[-1] if xray_data_plot else 0
                xray_data_plot.extend([last_val] * (min_len_ts - len(xray_data_plot)))
            elif len(xray_data_plot) > min_len_ts:
                xray_data_plot = xray_data_plot[:min_len_ts]
            plt.plot(plot_timestamps, xray_data_plot, label=f"X-ray Oda {room_idx + 1} Kuyruğu", alpha=0.8)

        plt.axvline(x=LUNCH_START, color='grey', linestyle=':', linewidth=1, label=f'Öğle Başlangıcı ({LUNCH_START})')
        plt.axvline(x=LUNCH_END, color='dimgrey', linestyle='--', linewidth=1.2, label=f'Öğle Bitişi ({LUNCH_END})')
        plt.axvline(x=WALKIN_CUTOFF_TIME, color='blue', linestyle='-.', linewidth=1.2,
                    label=f'Walk-in Kesim ({WALKIN_CUTOFF_TIME})')
        plt.axvline(x=SIM_TIME, color='red', linestyle='-', linewidth=1.5, label=f'Randevu Kesim ({SIM_TIME})')
        plt.xlabel("Zaman (dakika)");
        plt.ylabel("X-ray Oda Kuyruk Uzunluğu")
        plt.title(
            f"X-ray Oda Kuyruk Uzunlukları ({NUM_XRAY_ROOMS} Oda, ÖğleÖnce x1.25 Yavaş)\nTohum: {RANDOM_SEED}{title_suffix}")
        plt.legend(fontsize='small', loc='upper left');
        plt.grid(True, linestyle=':', alpha=0.7);
        plt.tight_layout();
        plt.show()

        #Consultant Queue Plot
        plt.figure(figsize=(14, 7))
        for consultant_idx in range(NUM_CONSULTANTS):
            consultant_data_plot = consultant_queue_lengths.get(consultant_idx, [])
            if len(consultant_data_plot) < min_len_ts:
                last_val = consultant_data_plot[-1] if consultant_data_plot else 0
                consultant_data_plot.extend([last_val] * (min_len_ts - len(consultant_data_plot)))
            elif len(consultant_data_plot) > min_len_ts:
                consultant_data_plot = consultant_data_plot[:min_len_ts]
            plt.plot(plot_timestamps, consultant_data_plot, label=f"Danışman {consultant_idx + 1} Kuyruğu", alpha=0.8)

        plt.axvline(x=LUNCH_START, color='grey', linestyle=':', linewidth=1, label=f'Öğle Başlangıcı ({LUNCH_START})')
        plt.axvline(x=LUNCH_END, color='dimgrey', linestyle='--', linewidth=1.2, label=f'Öğle Bitişi ({LUNCH_END})')
        plt.axvline(x=WALKIN_CUTOFF_TIME, color='blue', linestyle='-.', linewidth=1.2,
                    label=f'Walk-in Kesim ({WALKIN_CUTOFF_TIME})')
        plt.axvline(x=SIM_TIME, color='red', linestyle='-', linewidth=1.5, label=f'Randevu Kesim ({SIM_TIME})')
        plt.xlabel("Zaman (dakika)");
        plt.ylabel("Danışman Kuyruk Uzunluğu")
        plt.title(
            f"Danışman Kuyruk Uzunlukları (Randevusuz Hasta Kaydı - Uniform Hizmet Süresi)\nTohum: {RANDOM_SEED}{title_suffix}")
        plt.legend(fontsize='small', loc='upper left');
        plt.grid(True, linestyle=':', alpha=0.7);
        plt.tight_layout();
        plt.show()

    print("\n--- Hasta Akış İstatistikleri ---")
    print(f"Toplam Planlanan Randevu Sayısı (Tüm Doktorlar): {sum(len(s) for s in all_doctors_schedules)}")
    print(f"Toplam Üretilen Hasta (Process Başlatılan): {total_patients_generated}")
    print(f"  Randevulu Gelen (Gerçekleşen): {appointment_arrival_count}")
    print(f"  Randevusuz Gelen (Walk-in): {walk_in_arrival_count}")
    total_departures = appointment_departure_count + walk_in_departure_count
    print(f"\nToplam Sistemi Tamamlayan Hasta: {total_departures}")
    print(f"  Randevulu Ayrılan: {appointment_departure_count}")
    print(f"  Randevusuz Ayrılan: {walk_in_departure_count}")
    calculated_remaining = total_patients_generated - total_departures
    print(f"Simülasyon Sonunda Sistemde Kalan Hasta (Hesaplanan): {calculated_remaining}")
    print(f"Simülasyon Sonunda Sistemde Kalan Hasta (Sayaç): {patients_currently_in_system}")
    if calculated_remaining != patients_currently_in_system:
        print("!!! UYARI: Hesaplanan ve sayaçtaki kalan hasta sayısı farklı! Kontrol edin.")

    print("\n--- Doktor Aktivite İstatistikleri ---")
    print("Doktorlar Tarafından Bakılan Toplam Muayene Sayıları:")
    for i in range(NUM_DOCTORS):
        total_exams_by_doc = doctor_patient_count[i] + doctor_second_exam_count[i]
        print(
            f"  Doktor {i + 1}: {total_exams_by_doc} (1. Muayene: {doctor_patient_count[i]}, 2. Muayene: {doctor_second_exam_count[i]})")
    print(f"\nX-ray'e Giden Toplam Hasta Sayısı (Tüm Odalar): {xray_patient_count} hasta")

    print("\n--- Danışman Aktivite İstatistikleri ---")
    total_consultant_services = sum(consultant_patient_count)
    print(f"Danışmanlar Tarafından Kayıt Edilen Toplam Hasta Sayısı: {total_consultant_services}")
    for i in range(NUM_CONSULTANTS):
        print(f"  Danışman {i + 1}: {consultant_patient_count[i]} hasta")

    print("\n--- Doktor Bazlı X-ray Yönlendirme İstatistikleri ---")
    xray_column_header = "X-ray'e Giden"
    print(
        f"{'Doktor':<10} | {'Gelen Hasta':<12} | {xray_column_header:<14} | {'X-ray Oranı (%)':<16} | {'Beklenen Oran (%)':<18}")
    print("-" * 80)
    for i in range(NUM_DOCTORS):
        total_patients_for_doc = doctor_patient_count[i]
        patients_to_xray_from_doc = doctor_second_exam_count[i]

        xray_referral_rate = 0
        if total_patients_for_doc > 0:
            xray_referral_rate = (patients_to_xray_from_doc / total_patients_for_doc) * 100

        expected_xray_prob_percent = doctor_configs[i]["xray_probability"] * 100

        print(
            f"Dr {i + 1:<7} | {total_patients_for_doc:<12} | {patients_to_xray_from_doc:<14} | {xray_referral_rate:<16.2f} | {expected_xray_prob_percent:<18.2f}")

    total_first_exams_all_docs = sum(doctor_patient_count)
    total_second_exams_all_docs = sum(doctor_second_exam_count)

    if total_second_exams_all_docs != xray_patient_count:
        print(
            f"\nNot: Toplam ikinci muayene sayısı ({total_second_exams_all_docs}) ile X-ray hizmeti alan toplam hasta sayısı ({xray_patient_count}) farklı olabilir.")
        print(
            "Bu durum, hastaların X-ray'den sonra ikinci muayeneye giremeden sistemden ayrılması (örn. simülasyon sonu) veya X-ray sırasında kesintiye uğraması gibi nedenlerle oluşabilir.")

    overall_xray_rate = 0
    if total_first_exams_all_docs > 0:
        overall_xray_rate = (total_second_exams_all_docs / total_first_exams_all_docs) * 100

    print("-" * 80)
    print(
        f"{'Toplam':<10} | {total_first_exams_all_docs:<12} | {total_second_exams_all_docs:<14} | {overall_xray_rate:<16.2f} | {'N/A':<18}")

    for consultant_idx in range(NUM_CONSULTANTS):
        consultant_queue_data = consultant_queue_lengths.get(consultant_idx, [])
        descriptive_stats(consultant_queue_data, f"Danışman {consultant_idx + 1} Kuyruk Uzunluğu")


if __name__ == "__main__":
    for i in range(1):
        main()