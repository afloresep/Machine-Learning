class School:
  
  def __init__(self, name, level, numberOfStudent):
    self.name = name
    self.level = level
    self.numberOfStudent = numberOfStudent

  def get_name(self):
    return self.name

  def get_level(self):
    return self.level

  def get_numberOfStudents(self):
    return self.numberOfStudent

  def set_numberOfStudents(self, new_amount):
    self.numberOfStudents = new_amount

  def __repr__(self):
    return f"A {self.level} school name {self.name} with {self.numberOfStudent} students."




class PrimarySchool(School):
  def __init__(self, name, numberOfStudent, pickuppolicy):
    super().__init__(name, 'primary', numberOfStudent)
    self.pickuppolicy = pickuppolicy

  def get_policy(self):
    return self.pickuppolicy

  def __repr__(self):
    parentRepr = super().__repr__()
    return parentRepr + "The pickup polity is {pickupPolicy}".format(pickupPolicy = self.pickuppolicy)


mySchool = School("Codecademy", "high", 100)
print(mySchool)
print(mySchool.get_name())
print(mySchool.get_level())
mySchool.set_numberOfStudents(200)
print(mySchool.get_numberOfStudents())

print('\n')

testSchool = PrimarySchool("Codecademy", 300, "Pickup Allowed")
print(testSchool.get_policy())
print(testSchool)
print('\n')

class HighSchool(School):
  def __init__(self, name, numberOfStudent, sportsTeams):
    super().__init__(name, 'High', numberOfStudent)
    self.sportsTeams = sportsTeams

  def getSportsTeams(self):
    return self.sportsTeams
  def __repr__(self):
    parent = super().__repr__()
    return parent + f" information of our sport Team {self.sportsTeams}"

c = HighSchool("Codecademy High", 500, ["Tennis", "Basketball"])
print(c)
