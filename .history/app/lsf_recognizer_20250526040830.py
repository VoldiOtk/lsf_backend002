import cv2
import numpy as np
from .hand_detector import HandDetector

class LSFRecognizer:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.last_signs = []  # Pour stocker l'historique des signes
        self.max_history = 5  # Nombre maximum de signes à mémoriser
        # Dictionnaire des signes connus
        self.known_signs = {
            "bonjour": self._is_bonjour,
            "merci": self._is_merci,
            "oui": self._is_oui,
            "non": self._is_non,
            "au_revoir": self._is_au_revoir,
            "poing_ferme": self._is_poing_ferme,
            "s_il_vous_plait": self._is_s_il_vous_plait,
            "je_t_aime": self._is_je_t_aime,
            "bien": self._is_bien,
            "manger": self._is_manger,
            "aide": self._is_aide,
            "attendre": self._is_attendre,
            "comprendre": self._is_comprendre,
            "faim": self._is_faim,
            "fatigue": self._is_fatigue,
            "dormir": self._is_dormir,
            "boire": self._is_boire,
            "froid": self._is_froid,
            "chaud": self._is_chaud,
            "pardon": self._is_pardon,
            "aujourd_hui": self._is_aujourd_hui,
            "demain": self._is_demain,
            "bonne_nuit": self._is_bonne_nuit,
            "sante": self._is_sante,
            "amitie": self._is_amitie,
            "famille": self._is_famille,
            "ecole": self._is_ecole,
            "un": self._is_un,
            "deux": self._is_deux,
            "trois": self._is_trois,
            "quatre": self._is_quatre,
            "cinq": self._is_cinq,
            "soleil": self._is_soleil,
            "lune": self._is_lune,
            "etoile": self._is_etoile,
            "pluie": self._is_pluie,
            "neige": self._is_neige,
            "vent": self._is_vent,
            "feu": self._is_feu,
            "eau": self._is_eau,
            "terre": self._is_terre,
            "ciel": self._is_ciel
        }

    def recognize_sign(self, frame):
        """
        Reconnaît le signe LSF dans l'image
        """
        # Détecter la main
        frame_with_landmarks, landmarks = self.hand_detector.detect_hand(frame)
        
        if landmarks is None:
            return "Pas de main détectée"

        # Vérifier chaque signe connu
        detected_sign = None
        for sign_name, check_function in self.known_signs.items():
            if check_function(landmarks):
                detected_sign = sign_name
                break

        if detected_sign is None:
            detected_sign = "Signe non reconnu"

        # Mettre à jour l'historique des signes
        self.last_signs.append(detected_sign)
        if len(self.last_signs) > self.max_history:
            self.last_signs.pop(0)

        # Vérifier les phrases
        phrase = self._check_phrases()
        if phrase:
            return phrase

        return detected_sign

    def _check_phrases(self):
        """
        Vérifie si la séquence de signes forme une phrase connue
        """
        # Convertir la séquence en chaîne pour faciliter la recherche
        sequence = " ".join(self.last_signs)
        
        # Phrases courantes
        phrases = {
            "bonjour comment ca vas": ["bonjour", "comment", "ca", "vas"],
            "bonjour comment allez vous": ["bonjour", "comment", "allez", "vous"],
            "je m appelle": ["je", "m", "appelle"],
            "enchanté de vous rencontrer": ["enchanté", "de", "vous", "rencontrer"],
            "au revoir à bientôt": ["au_revoir", "à", "bientôt"],
            "merci beaucoup": ["merci", "beaucoup"],
            "s il vous plait": ["s_il_vous_plait"],
            "je t aime": ["je_t_aime"],
            "bonne nuit": ["bonne_nuit"],
            "bonne journée": ["bonne", "journée"],
            "à tout à l heure": ["à", "tout", "à", "l", "heure"],
            "je ne comprends pas": ["je", "ne", "comprends", "pas"],
            "pouvez vous répéter": ["pouvez", "vous", "répéter"],
            "je suis fatigué": ["je", "suis", "fatigue"],
            "je suis malade": ["je", "suis", "malade"],
            "je suis heureux": ["je", "suis", "heureux"],
            "je suis triste": ["je", "suis", "triste"],
            "je suis en colère": ["je", "suis", "en", "colere"],
            "je suis surpris": ["je", "suis", "surpris"],
            "je suis désolé": ["je", "suis", "desole"],
            "je suis perdu": ["je", "suis", "perdu"],
            "je suis pressé": ["je", "suis", "presse"],
            "je suis en retard": ["je", "suis", "en", "retard"],
            "je suis à l heure": ["je", "suis", "à", "l", "heure"],
            "je suis occupé": ["je", "suis", "occupe"],
            "je suis libre": ["je", "suis", "libre"],
            "je suis prêt": ["je", "suis", "pret"],
            "je suis là": ["je", "suis", "la"],
            "je suis parti": ["je", "suis", "parti"],
            "je suis revenu": ["je", "suis", "revenu"],
            "je suis arrivé": ["je", "suis", "arrive"],
            "je suis arrivé": ["je", "suis", "arrivé"],
            "je suis en train de": ["je", "suis", "en", "train", "de"],
            "je suis en train de manger": ["je", "suis", "en", "train", "de", "manger"],
            "je suis en train de boire": ["je", "suis", "en", "train", "de", "boire"],
            "je suis en train de dormir": ["je", "suis", "en", "train", "de", "dormir"],
            "je suis en train de travailler": ["je", "suis", "en", "train", "de", "travailler"],
            "je suis en train d apprendre": ["je", "suis", "en", "train", "d", "apprendre"],
            "je suis en train de comprendre": ["je", "suis", "en", "train", "de", "comprendre"],
            "je suis en train de réfléchir": ["je", "suis", "en", "train", "de", "réfléchir"],
            "je suis en train de parler": ["je", "suis", "en", "train", "de", "parler"],
            "je suis en train d écouter": ["je", "suis", "en", "train", "d", "écouter"],
            "je suis en train de regarder": ["je", "suis", "en", "train", "de", "regarder"],
            "je suis en train de chercher": ["je", "suis", "en", "train", "de", "chercher"],
            "je suis en train de trouver": ["je", "suis", "en", "train", "de", "trouver"],
            "je suis en train de perdre": ["je", "suis", "en", "train", "de", "perdre"],
            "je suis en train de gagner": ["je", "suis", "en", "train", "de", "gagner"],
            "je suis en train de jouer": ["je", "suis", "en", "train", "de", "jouer"],
            "je suis en train de gagner": ["je", "suis", "en", "train", "de", "gagner"],
            "je suis en train de perdre": ["je", "suis", "en", "train", "de", "perdre"],
            "je suis en train de gagner": ["je", "suis", "en", "train", "de", "gagner"],
            "je suis en train de perdre": ["je", "suis", "en", "train", "de", "perdre"],
            "je vais à l école": ["je", "vais", "à", "l", "ecole"],
            "je vais au travail": ["je", "vais", "au", "travail"],
            "je vais à la maison": ["je", "vais", "à", "la", "maison"],
            "je vais au magasin": ["je", "vais", "au", "magasin"],
            "je vais au restaurant": ["je", "vais", "au", "restaurant"],
            "je vais au cinéma": ["je", "vais", "au", "cinema"],
            "je vais au parc": ["je", "vais", "au", "parc"],
            "je vais à la plage": ["je", "vais", "à", "la", "plage"],
            "je vais à la montagne": ["je", "vais", "à", "la", "montagne"],
            "je vais à la campagne": ["je", "vais", "à", "la", "campagne"],
            "je vais à la ville": ["je", "vais", "à", "la", "ville"],
            "je vais à la gare": ["je", "vais", "à", "la", "gare"],
            "je vais à l aéroport": ["je", "vais", "à", "l", "aeroport"],
            "je vais à l hôpital": ["je", "vais", "à", "l", "hopital"],
            "je vais au docteur": ["je", "vais", "au", "docteur"],
            "je vais à la pharmacie": ["je", "vais", "à", "la", "pharmacie"],
            "je vais à la banque": ["je", "vais", "à", "la", "banque"],
            "je vais à la poste": ["je", "vais", "à", "la", "poste"],
            "je vais à la bibliothèque": ["je", "vais", "à", "la", "bibliotheque"],
            "je vais au musée": ["je", "vais", "au", "musee"],
            "je vais au théâtre": ["je", "vais", "au", "theatre"],
            "je vais au concert": ["je", "vais", "au", "concert"],
            "je vais au stade": ["je", "vais", "au", "stade"],
            "je vais à la piscine": ["je", "vais", "à", "la", "piscine"],
            "je vais au gymnase": ["je", "vais", "au", "gymnase"],
            "je vais à la salle de sport": ["je", "vais", "à", "la", "salle", "de", "sport"],
            "je vais à la salle de bain": ["je", "vais", "à", "la", "salle", "de", "bain"],
            "je vais à la cuisine": ["je", "vais", "à", "la", "cuisine"],
            "je vais au salon": ["je", "vais", "au", "salon"],
            "je vais à la chambre": ["je", "vais", "à", "la", "chambre"],
            "je vais au jardin": ["je", "vais", "au", "jardin"],
            "je vais au garage": ["je", "vais", "au", "garage"],
            "je vais au sous-sol": ["je", "vais", "au", "sous-sol"],
            "je vais au grenier": ["je", "vais", "au", "grenier"],
            "je vais au balcon": ["je", "vais", "au", "balcon"],
            "je vais à la terrasse": ["je", "vais", "à", "la", "terrasse"],
            "je vais à la cave": ["je", "vais", "à", "la", "cave"],
            "je vais à l ascenseur": ["je", "vais", "à", "l", "ascenseur"],
            "je vais à l escalier": ["je", "vais", "à", "l", "escalier"],
            "je vais à la porte": ["je", "vais", "à", "la", "porte"],
            "je vais à la fenêtre": ["je", "vais", "à", "la", "fenetre"],
            "je vais au toit": ["je", "vais", "au", "toit"],
            "je vais au mur": ["je", "vais", "au", "mur"],
            "je vais au plafond": ["je", "vais", "au", "plafond"],
            "je vais au sol": ["je", "vais", "au", "sol"],
            "je vais au coin": ["je", "vais", "au", "coin"],
            "je vais au centre": ["je", "vais", "au", "centre"],
            "je vais à côté": ["je", "vais", "à", "côté"],
            "je vais devant": ["je", "vais", "devant"],
            "je vais derrière": ["je", "vais", "derrière"],
            "je vais à gauche": ["je", "vais", "à", "gauche"],
            "je vais à droite": ["je", "vais", "à", "droite"],
            "je vais en haut": ["je", "vais", "en", "haut"],
            "je vais en bas": ["je", "vais", "en", "bas"],
            "je vais au milieu": ["je", "vais", "au", "milieu"],
            "je vais au début": ["je", "vais", "au", "début"],
            "je vais à la fin": ["je", "vais", "à", "la", "fin"],
            "je vais au début": ["je", "vais", "au", "début"],
            "je vais à la fin": ["je", "vais", "à", "la", "fin"]
        }

        # Vérifier chaque phrase
        for phrase, required_signs in phrases.items():
            if all(sign in self.last_signs for sign in required_signs):
                # Vérifier l'ordre des signes
                last_signs_str = " ".join(self.last_signs)
                if " ".join(required_signs) in last_signs_str:
                    return phrase

        return None

    def _is_bonjour(self, landmarks):
        """
        Vérifie si le signe est "Bonjour"
        """
        # Logique simplifiée : vérifie si l'index et le majeur sont levés
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        return index_tip.y < landmarks.landmark[6].y and middle_tip.y < landmarks.landmark[10].y

    def _is_merci(self, landmarks):
        """
        Vérifie si le signe est "Merci"
        """
        # Logique simplifiée : vérifie si le pouce est levé
        thumb_tip = landmarks.landmark[4]
        return thumb_tip.y < landmarks.landmark[3].y

    def _is_oui(self, landmarks):
        """
        Vérifie si le signe est "Oui"
        """
        # Logique simplifiée : vérifie si l'index est levé
        index_tip = landmarks.landmark[8]
        return index_tip.y < landmarks.landmark[6].y

    def _is_non(self, landmarks):
        """
        Vérifie si le signe est "Non"
        """
        # Logique simplifiée : vérifie si l'index fait un mouvement horizontal
        index_tip = landmarks.landmark[8]
        index_mcp = landmarks.landmark[5]
        return abs(index_tip.x - index_mcp.x) > 0.1

    def _is_au_revoir(self, landmarks):
        """
        Vérifie si le signe est "Au revoir"
        """
        # Logique simplifiée : vérifie si la main est ouverte
        pinky_tip = landmarks.landmark[20]
        pinky_mcp = landmarks.landmark[17]
        return pinky_tip.y < pinky_mcp.y

    def _is_poing_ferme(self, landmarks):
        """
        Vérifie si le signe est "Poing fermé"
        """
        # Logique simplifiée : vérifie si tous les doigts sont repliés
        finger_tips = [8, 12, 16, 20]  # Index, majeur, annulaire, auriculaire
        finger_mcps = [5, 9, 13, 17]   # Points de référence
        return all(landmarks.landmark[tip].y > landmarks.landmark[mcp].y 
                  for tip, mcp in zip(finger_tips, finger_mcps))

    def _is_s_il_vous_plait(self, landmarks):
        """
        Vérifie si le signe est "S'il vous plaît"
        """
        # Vérifie si la main est plate et tournée vers le haut
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.y < wrist.y and abs(middle_tip.x - wrist.x) < 0.1

    def _is_je_t_aime(self, landmarks):
        """
        Vérifie si le signe est "Je t'aime"
        """
        # Vérifie si l'index et le majeur sont croisés
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        return abs(index_tip.x - middle_tip.x) < 0.05 and abs(index_tip.y - middle_tip.y) < 0.05

    def _is_bien(self, landmarks):
        """
        Vérifie si le signe est "Bien"
        """
        # Vérifie si le pouce est levé et les autres doigts sont repliés
        thumb_tip = landmarks.landmark[4]
        thumb_mcp = landmarks.landmark[2]
        finger_tips = [8, 12, 16, 20]  # Index, majeur, annulaire, auriculaire
        finger_mcps = [5, 9, 13, 17]   # Points de référence
        return (thumb_tip.y < thumb_mcp.y and 
                all(landmarks.landmark[tip].y > landmarks.landmark[mcp].y 
                    for tip, mcp in zip(finger_tips, finger_mcps)))

    def _is_manger(self, landmarks):
        """
        Vérifie si le signe est "Manger"
        """
        # Vérifie si la main est près du visage
        wrist = landmarks.landmark[0]
        return wrist.y < 0.3  # La main doit être dans la partie supérieure de l'image

    def _is_aide(self, landmarks):
        """
        Vérifie si le signe est "Aide"
        """
        # Vérifie si la main est ouverte et tournée vers le haut
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]  # Index, majeur, annulaire, auriculaire
        return all(landmarks.landmark[tip].y < wrist.y for tip in finger_tips)

    def _is_attendre(self, landmarks):
        """
        Vérifie si le signe est "Attendre"
        """
        # Vérifie si la main est immobile (position neutre)
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return abs(middle_tip.x - wrist.x) < 0.05 and abs(middle_tip.y - wrist.y) < 0.05

    def _is_comprendre(self, landmarks):
        """
        Vérifie si le signe est "Comprendre"
        """
        # Vérifie si l'index est près du front
        index_tip = landmarks.landmark[8]
        return index_tip.y < 0.2  # La main doit être dans la partie supérieure de l'image

    def _is_faim(self, landmarks):
        """
        Vérifie si le signe est "Faim"
        """
        # Vérifie si la main est dans la partie centrale de l'image
        wrist = landmarks.landmark[0]
        return 0.3 < wrist.y < 0.7  # La main doit être dans la partie centrale

    def _is_fatigue(self, landmarks):
        """
        Vérifie si le signe est "Fatigué"
        """
        # Vérifie si la main est près du front
        wrist = landmarks.landmark[0]
        return wrist.y < 0.25  # La main doit être dans la partie supérieure de l'image

    def _is_dormir(self, landmarks):
        """
        Vérifie si le signe est "Dormir"
        """
        # Vérifie si la main est près de la joue
        wrist = landmarks.landmark[0]
        return wrist.y < 0.3 and abs(wrist.x - 0.5) < 0.2  # Main près du visage

    def _is_boire(self, landmarks):
        """
        Vérifie si le signe est "Boire"
        """
        # Vérifie si la main est près de la bouche
        wrist = landmarks.landmark[0]
        return wrist.y < 0.35 and abs(wrist.x - 0.5) < 0.15  # Main près de la bouche

    def _is_froid(self, landmarks):
        """
        Vérifie si le signe est "Froid"
        """
        # Vérifie si les doigts sont légèrement écartés
        finger_tips = [8, 12, 16, 20]  # Index, majeur, annulaire, auriculaire
        return all(abs(landmarks.landmark[tip].x - landmarks.landmark[tip-1].x) > 0.05 
                  for tip in finger_tips)

    def _is_chaud(self, landmarks):
        """
        Vérifie si le signe est "Chaud"
        """
        # Vérifie si la main est ouverte et les doigts sont écartés
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return all(landmarks.landmark[tip].y < wrist.y and 
                  abs(landmarks.landmark[tip].x - landmarks.landmark[tip-1].x) > 0.1 
                  for tip in finger_tips)

    def _is_pardon(self, landmarks):
        """
        Vérifie si le signe est "Pardon"
        """
        # Vérifie si la main fait un mouvement circulaire
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return abs(middle_tip.x - wrist.x) > 0.1 and abs(middle_tip.y - wrist.y) > 0.1

    def _is_aujourd_hui(self, landmarks):
        """
        Vérifie si le signe est "Aujourd'hui"
        """
        # Vérifie si l'index pointe vers le bas
        index_tip = landmarks.landmark[8]
        index_mcp = landmarks.landmark[5]
        return index_tip.y > index_mcp.y and abs(index_tip.x - index_mcp.x) < 0.05

    def _is_demain(self, landmarks):
        """
        Vérifie si le signe est "Demain"
        """
        # Vérifie si l'index pointe vers l'avant
        index_tip = landmarks.landmark[8]
        index_mcp = landmarks.landmark[5]
        return abs(index_tip.y - index_mcp.y) < 0.05 and index_tip.x > index_mcp.x

    def _is_bonne_nuit(self, landmarks):
        """
        Vérifie si le signe est "Bonne nuit"
        """
        # Vérifie si la main est près du visage et les doigts sont légèrement écartés
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]  # Index, majeur, annulaire, auriculaire
        return (wrist.y < 0.3 and  # Main près du visage
                all(0.05 < abs(landmarks.landmark[tip].x - landmarks.landmark[tip-1].x) < 0.15 
                    for tip in finger_tips))

    def _is_sante(self, landmarks):
        """
        Vérifie si le signe est "Santé"
        """
        # Vérifie si la main est levée et les doigts sont écartés
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(landmarks.landmark[tip].y < wrist.y and 
                    abs(landmarks.landmark[tip].x - landmarks.landmark[tip-1].x) > 0.1 
                    for tip in finger_tips))

    def _is_amitie(self, landmarks):
        """
        Vérifie si le signe est "Amitié"
        """
        # Vérifie si l'index et le majeur sont croisés
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        return (abs(index_tip.x - middle_tip.x) < 0.05 and 
                abs(index_tip.y - middle_tip.y) < 0.05 and
                index_tip.y < landmarks.landmark[6].y)  # Doigts levés

    def _is_famille(self, landmarks):
        """
        Vérifie si le signe est "Famille"
        """
        # Vérifie si la main est ouverte et les doigts sont légèrement écartés
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.5 and  # Main dans la partie supérieure
                all(0.05 < abs(landmarks.landmark[tip].x - landmarks.landmark[tip-1].x) < 0.1 
                    for tip in finger_tips))

    def _is_ecole(self, landmarks):
        """
        Vérifie si le signe est "École"
        """
        # Vérifie si la main est plate et tournée vers le haut
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(abs(landmarks.landmark[tip].y - wrist.y) < 0.1 
                    for tip in finger_tips))

    def _is_un(self, landmarks):
        """
        Vérifie si le signe est "Un"
        """
        # Vérifie si seul l'index est levé
        index_tip = landmarks.landmark[8]
        other_tips = [12, 16, 20]  # Majeur, annulaire, auriculaire
        return (index_tip.y < landmarks.landmark[6].y and  # Index levé
                all(landmarks.landmark[tip].y > landmarks.landmark[tip-2].y 
                    for tip in other_tips))  # Autres doigts baissés

    def _is_deux(self, landmarks):
        """
        Vérifie si le signe est "Deux"
        """
        # Vérifie si l'index et le majeur sont levés
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        other_tips = [16, 20]  # Annulaire, auriculaire
        return (index_tip.y < landmarks.landmark[6].y and  # Index levé
                middle_tip.y < landmarks.landmark[10].y and  # Majeur levé
                all(landmarks.landmark[tip].y > landmarks.landmark[tip-2].y 
                    for tip in other_tips))  # Autres doigts baissés

    def _is_trois(self, landmarks):
        """
        Vérifie si le signe est "Trois"
        """
        # Vérifie si l'index, le majeur et l'annulaire sont levés
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]
        return (index_tip.y < landmarks.landmark[6].y and  # Index levé
                middle_tip.y < landmarks.landmark[10].y and  # Majeur levé
                ring_tip.y < landmarks.landmark[14].y and  # Annulaire levé
                pinky_tip.y > landmarks.landmark[18].y)  # Auriculaire baissé

    def _is_quatre(self, landmarks):
        """
        Vérifie si le signe est "Quatre"
        """
        # Vérifie si tous les doigts sauf le pouce sont levés
        finger_tips = [8, 12, 16, 20]  # Index, majeur, annulaire, auriculaire
        finger_mcps = [5, 9, 13, 17]   # Points de référence
        return all(landmarks.landmark[tip].y < landmarks.landmark[mcp].y 
                  for tip, mcp in zip(finger_tips, finger_mcps))

    def _is_cinq(self, landmarks):
        """
        Vérifie si le signe est "Cinq"
        """
        # Vérifie si tous les doigts sont écartés
        finger_tips = [8, 12, 16, 20]  # Index, majeur, annulaire, auriculaire
        return all(abs(landmarks.landmark[tip].x - landmarks.landmark[tip-1].x) > 0.1 
                  for tip in finger_tips)

    def _is_soleil(self, landmarks):
        """
        Vérifie si le signe est "Soleil"
        """
        # Vérifie si la main est ouverte et les doigts sont écartés vers le haut
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.3 and  # Main levée
                all(landmarks.landmark[tip].y < wrist.y and 
                    abs(landmarks.landmark[tip].x - landmarks.landmark[tip-1].x) > 0.15 
                    for tip in finger_tips))

    def _is_lune(self, landmarks):
        """
        Vérifie si le signe est "Lune"
        """
        # Vérifie si l'index et le majeur forment un croissant
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        return (abs(index_tip.x - middle_tip.x) > 0.2 and  # Doigts écartés horizontalement
                abs(index_tip.y - middle_tip.y) < 0.1)  # Même hauteur

    def _is_etoile(self, landmarks):
        """
        Vérifie si le signe est "Étoile"
        """
        # Vérifie si tous les doigts sont écartés en étoile
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(abs(landmarks.landmark[tip].x - wrist.x) > 0.2 and 
                    abs(landmarks.landmark[tip].y - wrist.y) > 0.2 
                    for tip in finger_tips))

    def _is_pluie(self, landmarks):
        """
        Vérifie si le signe est "Pluie"
        """
        # Vérifie si les doigts pointent vers le bas
        finger_tips = [8, 12, 16, 20]
        finger_mcps = [5, 9, 13, 17]
        return all(landmarks.landmark[tip].y > landmarks.landmark[mcp].y 
                  for tip, mcp in zip(finger_tips, finger_mcps))

    def _is_neige(self, landmarks):
        """
        Vérifie si le signe est "Neige"
        """
        # Vérifie si les doigts sont écartés et pointent vers le bas
        finger_tips = [8, 12, 16, 20]
        return all(abs(landmarks.landmark[tip].x - landmarks.landmark[tip-1].x) > 0.1 and
                  landmarks.landmark[tip].y > landmarks.landmark[tip-2].y 
                  for tip in finger_tips)

    def _is_vent(self, landmarks):
        """
        Vérifie si le signe est "Vent"
        """
        # Vérifie si la main est horizontale et les doigts sont écartés
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (abs(wrist.y - landmarks.landmark[8].y) < 0.1 and  # Main horizontale
                all(abs(landmarks.landmark[tip].x - landmarks.landmark[tip-1].x) > 0.15 
                    for tip in finger_tips))

    def _is_feu(self, landmarks):
        """
        Vérifie si le signe est "Feu"
        """
        # Vérifie si les doigts sont écartés vers le haut
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(landmarks.landmark[tip].y < wrist.y and 
                    abs(landmarks.landmark[tip].x - landmarks.landmark[tip-1].x) > 0.1 
                    for tip in finger_tips))

    def _is_eau(self, landmarks):
        """
        Vérifie si le signe est "Eau"
        """
        # Vérifie si la main est plate et fait un mouvement ondulant
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (abs(wrist.y - landmarks.landmark[8].y) < 0.1 and  # Main horizontale
                all(abs(landmarks.landmark[tip].y - wrist.y) < 0.1 
                    for tip in finger_tips))

    def _is_terre(self, landmarks):
        """
        Vérifie si le signe est "Terre"
        """
        # Vérifie si la main est plate et tournée vers le bas
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y > 0.6 and  # Main baissée
                all(abs(landmarks.landmark[tip].y - wrist.y) < 0.1 
                    for tip in finger_tips))

    def _is_ciel(self, landmarks):
        """
        Vérifie si le signe est "Ciel"
        """
        # Vérifie si la main est levée et les doigts sont écartés vers le haut
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.3 and  # Main très levée
                all(landmarks.landmark[tip].y < wrist.y and 
                    abs(landmarks.landmark[tip].x - landmarks.landmark[tip-1].x) > 0.1 
                    for tip in finger_tips))

    def _is_comment(self, landmarks):
        """
        Vérifie si le signe est "Comment"
        """
        # Vérifie si l'index fait un mouvement circulaire
        index_tip = landmarks.landmark[8]
        index_mcp = landmarks.landmark[5]
        return abs(index_tip.x - index_mcp.x) > 0.1 and abs(index_tip.y - index_mcp.y) > 0.1

    def _is_ca(self, landmarks):
        """
        Vérifie si le signe est "Ça"
        """
        # Vérifie si la main est plate et fait un mouvement de va-et-vient
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return abs(middle_tip.x - wrist.x) > 0.1

    def _is_vas(self, landmarks):
        """
        Vérifie si le signe est "Vas"
        """
        # Vérifie si l'index pointe vers l'avant
        index_tip = landmarks.landmark[8]
        index_mcp = landmarks.landmark[5]
        return index_tip.x > index_mcp.x and abs(index_tip.y - index_mcp.y) < 0.1

    def _is_je(self, landmarks):
        """
        Vérifie si le signe est "Je"
        """
        # Vérifie si l'index pointe vers soi
        index_tip = landmarks.landmark[8]
        return index_tip.x < 0.3  # Main du côté gauche

    def _is_suis(self, landmarks):
        """
        Vérifie si le signe est "Suis"
        """
        # Vérifie si la main est plate et fait un mouvement vers le bas
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.y > wrist.y

    def _is_il(self, landmarks):
        """
        Vérifie si le signe est "Il"
        """
        # Vérifie si l'index pointe vers l'extérieur
        index_tip = landmarks.landmark[8]
        return index_tip.x > 0.7  # Main du côté droit

    def _is_fait(self, landmarks):
        """
        Vérifie si le signe est "Fait"
        """
        # Vérifie si la main est plate et fait un mouvement vers l'avant
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.x > wrist.x

    def _is_beau(self, landmarks):
        """
        Vérifie si le signe est "Beau"
        """
        # Vérifie si la main est ouverte et tournée vers le haut
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(landmarks.landmark[tip].y < wrist.y 
                    for tip in finger_tips))

    def _is_pleut(self, landmarks):
        """
        Vérifie si le signe est "Pleut"
        """
        # Vérifie si les doigts pointent vers le bas et font un mouvement de va-et-vient
        finger_tips = [8, 12, 16, 20]
        finger_mcps = [5, 9, 13, 17]
        return all(landmarks.landmark[tip].y > landmarks.landmark[mcp].y 
                  for tip, mcp in zip(finger_tips, finger_mcps))

    def _is_quel(self, landmarks):
        """
        Vérifie si le signe est "Quel"
        """
        # Vérifie si l'index et le majeur sont levés et font un mouvement de question
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        return (index_tip.y < landmarks.landmark[6].y and 
                middle_tip.y < landmarks.landmark[10].y and
                abs(index_tip.x - middle_tip.x) < 0.1)

    def _is_votre(self, landmarks):
        """
        Vérifie si le signe est "Votre"
        """
        # Vérifie si la main est ouverte et pointe vers l'extérieur
        wrist = landmarks.landmark[0]
        return wrist.x > 0.5  # Main du côté droit

    def _is_nom(self, landmarks):
        """
        Vérifie si le signe est "Nom"
        """
        # Vérifie si l'index et le majeur sont croisés
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        return (abs(index_tip.x - middle_tip.x) < 0.05 and 
                abs(index_tip.y - middle_tip.y) < 0.05)

    def _is_content(self, landmarks):
        """
        Vérifie si le signe est "Content"
        """
        # Vérifie si la main est ouverte et fait un mouvement vers le haut
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(landmarks.landmark[tip].y < wrist.y 
                    for tip in finger_tips))

    def _is_triste(self, landmarks):
        """
        Vérifie si le signe est "Triste"
        """
        # Vérifie si la main est baissée et les doigts sont repliés
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y > 0.6 and  # Main baissée
                all(landmarks.landmark[tip].y > wrist.y 
                    for tip in finger_tips))

    def _is_colere(self, landmarks):
        """
        Vérifie si le signe est "Colère"
        """
        # Vérifie si la main est fermée et fait un mouvement vers l'avant
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(landmarks.landmark[tip].y > wrist.y 
                    for tip in finger_tips))

    def _is_surpris(self, landmarks):
        """
        Vérifie si le signe est "Surpris"
        """
        # Vérifie si tous les doigts sont écartés
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(abs(landmarks.landmark[tip].x - landmarks.landmark[tip-1].x) > 0.15 
                    for tip in finger_tips))

    def _is_malade(self, landmarks):
        """
        Vérifie si le signe est "Malade"
        """
        # Vérifie si la main est près du front
        wrist = landmarks.landmark[0]
        return wrist.y < 0.3  # Main près du front

    def _is_heureux(self, landmarks):
        """
        Vérifie si le signe est "Heureux"
        """
        # Vérifie si la main est ouverte et fait un mouvement circulaire
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (wrist.y < 0.4 and  # Main levée
                abs(middle_tip.x - wrist.x) > 0.1 and 
                abs(middle_tip.y - wrist.y) > 0.1)

    def _is_desole(self, landmarks):
        """
        Vérifie si le signe est "Désolé"
        """
        # Vérifie si la main est plate et fait un mouvement vers le bas
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.y > wrist.y

    def _is_perdu(self, landmarks):
        """
        Vérifie si le signe est "Perdu"
        """
        # Vérifie si la main fait un mouvement de recherche
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return abs(middle_tip.x - wrist.x) > 0.2

    def _is_presse(self, landmarks):
        """
        Vérifie si le signe est "Pressé"
        """
        # Vérifie si la main fait un mouvement rapide
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return abs(middle_tip.x - wrist.x) > 0.15

    def _is_retard(self, landmarks):
        """
        Vérifie si le signe est "Retard"
        """
        # Vérifie si la main pointe vers le bas
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.y > wrist.y

    def _is_heure(self, landmarks):
        """
        Vérifie si le signe est "Heure"
        """
        # Vérifie si la main pointe vers la montre
        wrist = landmarks.landmark[0]
        return wrist.x > 0.5  # Main du côté droit

    def _is_occupe(self, landmarks):
        """
        Vérifie si le signe est "Occupé"
        """
        # Vérifie si la main est fermée
        finger_tips = [8, 12, 16, 20]
        finger_mcps = [5, 9, 13, 17]
        return all(landmarks.landmark[tip].y > landmarks.landmark[mcp].y 
                  for tip, mcp in zip(finger_tips, finger_mcps))

    def _is_libre(self, landmarks):
        """
        Vérifie si le signe est "Libre"
        """
        # Vérifie si la main est ouverte
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return all(landmarks.landmark[tip].y < wrist.y 
                  for tip in finger_tips)

    def _is_pret(self, landmarks):
        """
        Vérifie si le signe est "Prêt"
        """
        # Vérifie si la main est ouverte et tournée vers l'avant
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.x > wrist.x

    def _is_la(self, landmarks):
        """
        Vérifie si le signe est "Là"
        """
        # Vérifie si l'index pointe vers le bas
        index_tip = landmarks.landmark[8]
        return index_tip.y > landmarks.landmark[6].y

    def _is_parti(self, landmarks):
        """
        Vérifie si le signe est "Parti"
        """
        # Vérifie si la main fait un mouvement vers l'extérieur
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.x > wrist.x

    def _is_revenu(self, landmarks):
        """
        Vérifie si le signe est "Revenu"
        """
        # Vérifie si la main fait un mouvement vers l'intérieur
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.x < wrist.x

    def _is_arrive(self, landmarks):
        """
        Vérifie si le signe est "Arrivé"
        """
        # Vérifie si la main est plate et fait un mouvement vers le bas
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.y > wrist.y

    def _is_train(self, landmarks):
        """
        Vérifie si le signe est "Train"
        """
        # Vérifie si la main fait un mouvement circulaire
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (abs(middle_tip.x - wrist.x) > 0.1 and 
                abs(middle_tip.y - wrist.y) > 0.1)

    def _is_apprendre(self, landmarks):
        """
        Vérifie si le signe est "Apprendre"
        """
        # Vérifie si la main est près du front
        wrist = landmarks.landmark[0]
        return wrist.y < 0.3  # Main près du front

    def _is_reflechir(self, landmarks):
        """
        Vérifie si le signe est "Réfléchir"
        """
        # Vérifie si l'index est près du front
        index_tip = landmarks.landmark[8]
        return index_tip.y < 0.2  # Index près du front

    def _is_parler(self, landmarks):
        """
        Vérifie si le signe est "Parler"
        """
        # Vérifie si la main est près de la bouche
        wrist = landmarks.landmark[0]
        return wrist.y < 0.35  # Main près de la bouche

    def _is_ecouter(self, landmarks):
        """
        Vérifie si le signe est "Écouter"
        """
        # Vérifie si la main est près de l'oreille
        wrist = landmarks.landmark[0]
        return wrist.y < 0.3 and wrist.x > 0.7  # Main près de l'oreille droite

    def _is_regarder(self, landmarks):
        """
        Vérifie si le signe est "Regarder"
        """
        # Vérifie si l'index et le majeur pointent vers l'avant
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        return (index_tip.x > landmarks.landmark[5].x and 
                middle_tip.x > landmarks.landmark[9].x)

    def _is_chercher(self, landmarks):
        """
        Vérifie si le signe est "Chercher"
        """
        # Vérifie si la main fait un mouvement de recherche
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return abs(middle_tip.x - wrist.x) > 0.2

    def _is_trouver(self, landmarks):
        """
        Vérifie si le signe est "Trouver"
        """
        # Vérifie si l'index pointe vers l'avant
        index_tip = landmarks.landmark[8]
        return index_tip.x > landmarks.landmark[5].x

    def _is_perdre(self, landmarks):
        """
        Vérifie si le signe est "Perdre"
        """
        # Vérifie si la main fait un mouvement vers le bas
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.y > wrist.y

    def _is_gagner(self, landmarks):
        """
        Vérifie si le signe est "Gagner"
        """
        # Vérifie si la main est levée
        wrist = landmarks.landmark[0]
        return wrist.y < 0.4  # Main levée

    def _is_jouer(self, landmarks):
        """
        Vérifie si le signe est "Jouer"
        """
        # Vérifie si la main fait un mouvement de jeu
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (abs(middle_tip.x - wrist.x) > 0.1 and 
                abs(middle_tip.y - wrist.y) > 0.1)

    def _is_travail(self, landmarks):
        """
        Vérifie si le signe est "Travail"
        """
        # Vérifie si la main fait un mouvement de travail
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (wrist.y < 0.4 and  # Main levée
                abs(middle_tip.x - wrist.x) > 0.1)

    def _is_maison(self, landmarks):
        """
        Vérifie si le signe est "Maison"
        """
        # Vérifie si la main forme un toit
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(landmarks.landmark[tip].y < wrist.y 
                    for tip in finger_tips))

    def _is_magasin(self, landmarks):
        """
        Vérifie si le signe est "Magasin"
        """
        # Vérifie si la main fait un mouvement d'ouverture
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.x > wrist.x

    def _is_restaurant(self, landmarks):
        """
        Vérifie si le signe est "Restaurant"
        """
        # Vérifie si la main est près de la bouche
        wrist = landmarks.landmark[0]
        return wrist.y < 0.35  # Main près de la bouche

    def _is_cinema(self, landmarks):
        """
        Vérifie si le signe est "Cinéma"
        """
        # Vérifie si la main forme un cadre
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(abs(landmarks.landmark[tip].x - wrist.x) > 0.1 
                    for tip in finger_tips))

    def _is_parc(self, landmarks):
        """
        Vérifie si le signe est "Parc"
        """
        # Vérifie si la main fait un mouvement circulaire
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (abs(middle_tip.x - wrist.x) > 0.1 and 
                abs(middle_tip.y - wrist.y) > 0.1)

    def _is_plage(self, landmarks):
        """
        Vérifie si le signe est "Plage"
        """
        # Vérifie si la main est plate et fait un mouvement horizontal
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return abs(middle_tip.x - wrist.x) > 0.2

    def _is_montagne(self, landmarks):
        """
        Vérifie si le signe est "Montagne"
        """
        # Vérifie si la main forme un pic
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (wrist.y < 0.4 and  # Main levée
                middle_tip.y < wrist.y)

    def _is_campagne(self, landmarks):
        """
        Vérifie si le signe est "Campagne"
        """
        # Vérifie si la main fait un mouvement ondulant
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (abs(middle_tip.x - wrist.x) > 0.1 and 
                abs(middle_tip.y - wrist.y) > 0.1)

    def _is_ville(self, landmarks):
        """
        Vérifie si le signe est "Ville"
        """
        # Vérifie si la main forme des bâtiments
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(landmarks.landmark[tip].y < wrist.y 
                    for tip in finger_tips))

    def _is_gare(self, landmarks):
        """
        Vérifie si le signe est "Gare"
        """
        # Vérifie si la main fait un mouvement de train
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (abs(middle_tip.x - wrist.x) > 0.2 and 
                abs(middle_tip.y - wrist.y) < 0.1)

    def _is_aeroport(self, landmarks):
        """
        Vérifie si le signe est "Aéroport"
        """
        # Vérifie si la main fait un mouvement d'avion
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (wrist.y < 0.4 and  # Main levée
                middle_tip.x > wrist.x)

    def _is_hopital(self, landmarks):
        """
        Vérifie si le signe est "Hôpital"
        """
        # Vérifie si la main forme une croix
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(abs(landmarks.landmark[tip].x - wrist.x) > 0.1 
                    for tip in finger_tips))

    def _is_docteur(self, landmarks):
        """
        Vérifie si le signe est "Docteur"
        """
        # Vérifie si la main est près du front
        wrist = landmarks.landmark[0]
        return wrist.y < 0.3  # Main près du front

    def _is_pharmacie(self, landmarks):
        """
        Vérifie si le signe est "Pharmacie"
        """
        # Vérifie si la main forme une croix
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(abs(landmarks.landmark[tip].x - wrist.x) > 0.1 
                    for tip in finger_tips))

    def _is_banque(self, landmarks):
        """
        Vérifie si le signe est "Banque"
        """
        # Vérifie si la main fait un mouvement d'argent
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (abs(middle_tip.x - wrist.x) > 0.1 and 
                abs(middle_tip.y - wrist.y) < 0.1)

    def _is_poste(self, landmarks):
        """
        Vérifie si le signe est "Poste"
        """
        # Vérifie si la main fait un mouvement d'enveloppe
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (wrist.y < 0.4 and  # Main levée
                middle_tip.x > wrist.x)

    def _is_bibliotheque(self, landmarks):
        """
        Vérifie si le signe est "Bibliothèque"
        """
        # Vérifie si la main fait un mouvement de livre
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (wrist.y < 0.4 and  # Main levée
                abs(middle_tip.x - wrist.x) < 0.1)

    def _is_musee(self, landmarks):
        """
        Vérifie si le signe est "Musée"
        """
        # Vérifie si la main fait un mouvement de tableau
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (wrist.y < 0.4 and  # Main levée
                abs(middle_tip.x - wrist.x) > 0.1)

    def _is_theatre(self, landmarks):
        """
        Vérifie si le signe est "Théâtre"
        """
        # Vérifie si la main fait un mouvement de rideau
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (wrist.y < 0.4 and  # Main levée
                middle_tip.y < wrist.y)

    def _is_concert(self, landmarks):
        """
        Vérifie si le signe est "Concert"
        """
        # Vérifie si la main fait un mouvement de musique
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (abs(middle_tip.x - wrist.x) > 0.1 and 
                abs(middle_tip.y - wrist.y) > 0.1)

    def _is_stade(self, landmarks):
        """
        Vérifie si le signe est "Stade"
        """
        # Vérifie si la main forme un ovale
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(abs(landmarks.landmark[tip].x - wrist.x) > 0.1 
                    for tip in finger_tips))

    def _is_piscine(self, landmarks):
        """
        Vérifie si le signe est "Piscine"
        """
        # Vérifie si la main fait un mouvement de nage
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (abs(middle_tip.x - wrist.x) > 0.2 and 
                abs(middle_tip.y - wrist.y) < 0.1)

    def _is_gymnase(self, landmarks):
        """
        Vérifie si le signe est "Gymnase"
        """
        # Vérifie si la main fait un mouvement d'exercice
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (wrist.y < 0.4 and  # Main levée
                abs(middle_tip.x - wrist.x) > 0.1)

    def _is_salle(self, landmarks):
        """
        Vérifie si le signe est "Salle"
        """
        # Vérifie si la main forme un rectangle
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(abs(landmarks.landmark[tip].x - wrist.x) > 0.1 
                    for tip in finger_tips))

    def _is_sport(self, landmarks):
        """
        Vérifie si le signe est "Sport"
        """
        # Vérifie si la main fait un mouvement d'activité
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (wrist.y < 0.4 and  # Main levée
                abs(middle_tip.x - wrist.x) > 0.1)

    def _is_bain(self, landmarks):
        """
        Vérifie si le signe est "Bain"
        """
        # Vérifie si la main fait un mouvement de lavage
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (abs(middle_tip.x - wrist.x) > 0.1 and 
                abs(middle_tip.y - wrist.y) > 0.1)

    def _is_cuisine(self, landmarks):
        """
        Vérifie si le signe est "Cuisine"
        """
        # Vérifie si la main fait un mouvement de cuisson
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (wrist.y < 0.4 and  # Main levée
                abs(middle_tip.x - wrist.x) > 0.1)

    def _is_salon(self, landmarks):
        """
        Vérifie si le signe est "Salon"
        """
        # Vérifie si la main fait un mouvement de confort
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (wrist.y < 0.4 and  # Main levée
                abs(middle_tip.x - wrist.x) < 0.1)

    def _is_chambre(self, landmarks):
        """
        Vérifie si le signe est "Chambre"
        """
        # Vérifie si la main fait un mouvement de lit
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (wrist.y < 0.4 and  # Main levée
                middle_tip.y < wrist.y)

    def _is_jardin(self, landmarks):
        """
        Vérifie si le signe est "Jardin"
        """
        # Vérifie si la main fait un mouvement de plante
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (wrist.y < 0.4 and  # Main levée
                middle_tip.y < wrist.y)

    def _is_garage(self, landmarks):
        """
        Vérifie si le signe est "Garage"
        """
        # Vérifie si la main fait un mouvement de voiture
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (abs(middle_tip.x - wrist.x) > 0.2 and 
                abs(middle_tip.y - wrist.y) < 0.1)

    def _is_sous_sol(self, landmarks):
        """
        Vérifie si le signe est "Sous-sol"
        """
        # Vérifie si la main pointe vers le bas
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.y > wrist.y

    def _is_grenier(self, landmarks):
        """
        Vérifie si le signe est "Grenier"
        """
        # Vérifie si la main pointe vers le haut
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.y < wrist.y

    def _is_balcon(self, landmarks):
        """
        Vérifie si le signe est "Balcon"
        """
        # Vérifie si la main forme une plateforme
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(landmarks.landmark[tip].y < wrist.y 
                    for tip in finger_tips))

    def _is_terrasse(self, landmarks):
        """
        Vérifie si le signe est "Terrasse"
        """
        # Vérifie si la main forme une plateforme
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(landmarks.landmark[tip].y < wrist.y 
                    for tip in finger_tips))

    def _is_cave(self, landmarks):
        """
        Vérifie si le signe est "Cave"
        """
        # Vérifie si la main pointe vers le bas
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.y > wrist.y

    def _is_ascenseur(self, landmarks):
        """
        Vérifie si le signe est "Ascenseur"
        """
        # Vérifie si la main fait un mouvement vertical
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return abs(middle_tip.y - wrist.y) > 0.2

    def _is_escalier(self, landmarks):
        """
        Vérifie si le signe est "Escalier"
        """
        # Vérifie si la main fait un mouvement d'escalier
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (abs(middle_tip.x - wrist.x) > 0.1 and 
                abs(middle_tip.y - wrist.y) > 0.1)

    def _is_porte(self, landmarks):
        """
        Vérifie si le signe est "Porte"
        """
        # Vérifie si la main fait un mouvement d'ouverture
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.x > wrist.x

    def _is_fenetre(self, landmarks):
        """
        Vérifie si le signe est "Fenêtre"
        """
        # Vérifie si la main forme un cadre
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(abs(landmarks.landmark[tip].x - wrist.x) > 0.1 
                    for tip in finger_tips))

    def _is_toit(self, landmarks):
        """
        Vérifie si le signe est "Toit"
        """
        # Vérifie si la main forme un toit
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(landmarks.landmark[tip].y < wrist.y 
                    for tip in finger_tips))

    def _is_mur(self, landmarks):
        """
        Vérifie si le signe est "Mur"
        """
        # Vérifie si la main est plate et verticale
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return abs(middle_tip.x - wrist.x) < 0.1

    def _is_plafond(self, landmarks):
        """
        Vérifie si le signe est "Plafond"
        """
        # Vérifie si la main est plate et horizontale
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return abs(middle_tip.y - wrist.y) < 0.1

    def _is_sol(self, landmarks):
        """
        Vérifie si le signe est "Sol"
        """
        # Vérifie si la main est plate et horizontale
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return abs(middle_tip.y - wrist.y) < 0.1

    def _is_coin(self, landmarks):
        """
        Vérifie si le signe est "Coin"
        """
        # Vérifie si la main forme un angle
        wrist = landmarks.landmark[0]
        finger_tips = [8, 12, 16, 20]
        return (wrist.y < 0.4 and  # Main levée
                all(landmarks.landmark[tip].x != wrist.x 
                    for tip in finger_tips))

    def _is_centre(self, landmarks):
        """
        Vérifie si le signe est "Centre"
        """
        # Vérifie si la main pointe vers le centre
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (abs(middle_tip.x - wrist.x) < 0.1 and 
                abs(middle_tip.y - wrist.y) < 0.1)

    def _is_cote(self, landmarks):
        """
        Vérifie si le signe est "Côté"
        """
        # Vérifie si la main pointe sur le côté
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return abs(middle_tip.x - wrist.x) > 0.2

    def _is_devant(self, landmarks):
        """
        Vérifie si le signe est "Devant"
        """
        # Vérifie si la main pointe vers l'avant
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.x > wrist.x

    def _is_derriere(self, landmarks):
        """
        Vérifie si le signe est "Derrière"
        """
        # Vérifie si la main pointe vers l'arrière
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.x < wrist.x

    def _is_gauche(self, landmarks):
        """
        Vérifie si le signe est "Gauche"
        """
        # Vérifie si la main pointe vers la gauche
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.x < wrist.x

    def _is_droite(self, landmarks):
        """
        Vérifie si le signe est "Droite"
        """
        # Vérifie si la main pointe vers la droite
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.x > wrist.x

    def _is_haut(self, landmarks):
        """
        Vérifie si le signe est "Haut"
        """
        # Vérifie si la main pointe vers le haut
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.y < wrist.y

    def _is_bas(self, landmarks):
        """
        Vérifie si le signe est "Bas"
        """
        # Vérifie si la main pointe vers le bas
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.y > wrist.y

    def _is_milieu(self, landmarks):
        """
        Vérifie si le signe est "Milieu"
        """
        # Vérifie si la main pointe vers le milieu
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return (abs(middle_tip.x - wrist.x) < 0.1 and 
                abs(middle_tip.y - wrist.y) < 0.1)

    def _is_debut(self, landmarks):
        """
        Vérifie si le signe est "Début"
        """
        # Vérifie si la main pointe vers le début
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.x < wrist.x

    def _is_fin(self, landmarks):
        """
        Vérifie si le signe est "Fin"
        """
        # Vérifie si la main pointe vers la fin
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        return middle_tip.x > wrist.x

# Instance globale du reconnaisseur
recognizer = LSFRecognizer()

def recognize_lsf_sign(frame):
    """
    Fonction utilitaire pour reconnaître un signe LSF
    """
    return recognizer.recognize_sign(frame) 