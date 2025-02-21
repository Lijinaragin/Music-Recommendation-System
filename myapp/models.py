from django.db import models

# Create your models here.

class logintbl(models.Model):
    username=models.CharField(max_length=100)
    password=models.CharField(max_length=200)
    type=models.CharField(max_length=100)

class booktbl(models.Model):
    bookname=models.CharField(max_length=100)
    category = models.CharField(max_length=100)
    genre = models.CharField(max_length=100)
    author= models.CharField(max_length=100)
    ebooklink = models.CharField(max_length=500)

class songtbl(models.Model):
    songname = models.CharField(max_length=100)
    file = models.FileField()
    type=models.CharField(max_length=20)

class resttbl(models.Model):
    restname = models.CharField(max_length=100)
    lattitude= models.FloatField()
    logitude=models.FloatField()
    category=models.CharField(max_length=100)


class usertbl(models.Model):
    LOGIN=models.ForeignKey(logintbl,on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    phone= models.CharField(max_length=100)

class feedbacktbl(models.Model):
    USER = models.ForeignKey(usertbl, on_delete=models.CASCADE)
    feedback= models.CharField(max_length=100)
    date= models.DateField()

class playlist(models.Model):
    Song = models.ForeignKey(songtbl, on_delete=models.CASCADE)
    date=models.CharField(max_length=100)
class favtbl(models.Model):
    book= models.ForeignKey(booktbl, on_delete=models.CASCADE)
    song=models.ForeignKey(songtbl, on_delete=models.CASCADE)

class bookrating(models.Model):
    USER = models.ForeignKey(usertbl, on_delete=models.CASCADE)
    book = models.ForeignKey(booktbl, on_delete=models.CASCADE)
    rating= models.FloatField()
    feedback= models.CharField(max_length=100)
    date= models.DateField()
