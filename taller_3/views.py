from django.shortcuts import render
from django.http import HttpResponse
from os.path import exists
from taller_3.services import get_possible_movies, get_recommendation_by_movies

def index(request):
	if (request.method == 'GET'):
		return render(request, 'taller_3/taller_3.html', {'movies': get_possible_movies()})
	else:
		artists = request.POST.getlist('artists')
		rankings = [request.POST.get(artist) for artist in artists]
		recommendations = get_recommendation_by_artists(artists, rankings)
		return render(request, 'taller_1/taller_1.html', {'recommendations': recommendations, 'artists': get_possible_artists()})

def notebook(request):
	return render(request, 'taller_1/notebook.html')